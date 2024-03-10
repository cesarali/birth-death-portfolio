import torch
import torchcde
import torchsde

def gradient_penalty(generated, real, call,encoder_forward=None):
    assert generated.shape == real.shape
    batch_size = generated.size(0)

    alpha = torch.rand(batch_size, *[1 for _ in range(real.ndimension() - 1)],
                       dtype=generated.dtype, device=generated.device)
    interpolated = alpha * real.detach() + (1 - alpha) * generated.detach()
    interpolated.requires_grad_(True)

    with torch.enable_grad():
        if encoder_forward is None:
            score_interpolated = call(interpolated)
        else:
            score_interpolated = call(interpolated,encoder_forward)

        penalty, = torch.autograd.grad(score_interpolated, interpolated,
                                       torch.ones_like(score_interpolated),
                                       create_graph=True,retain_graph=True)


    return penalty.norm(2, dim=-1).sub(1).pow(2).mean()



class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh):
        super(MLP, self).__init__()

        model = [torch.nn.Linear(in_size, mlp_size),
                 torch.nn.Softplus()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            ###################
            # Note the use of softplus activations: these are used for theoretical reasons regarding the smoothness of
            # the vector fields of our SDE. It's unclear how much it matters in practice.
            ###################
            model.append(torch.nn.Softplus())
        model.append(torch.nn.Linear(mlp_size, out_size))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)

###################
# We begin by defining the generator SDE.
# The choice of Ito vs Stratonovich, and the choice of different noise types, isn't very important. We happen to be
# using Stratonovich with general noise.
###################
class GeneratorFunc(torch.nn.Module):
    sde_type = 'stratonovich'
    noise_type = 'general'

    def __init__(self, noise_size, hidden_size, mlp_size, num_layers,conditional_hidden=0):
        super(GeneratorFunc, self).__init__()

        self._noise_size = noise_size
        self._hidden_size = hidden_size
        self._conditional_hidden = conditional_hidden
        ###################
        # Drift and diffusion are MLPs. They happen to be the same size.
        # Note the final tanh nonlinearity: this is typically important for good performance, to constrain the rate of
        # change of the hidden state.
        ###################
        self._drift = MLP(1 + hidden_size + conditional_hidden, hidden_size, mlp_size, num_layers, tanh=True)
        self._diffusion = MLP(1 + hidden_size + conditional_hidden, hidden_size * noise_size, mlp_size, num_layers, tanh=True)
        self.condition_vector = None

    def f(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        t = t.expand(x.size(0), 1)
        if self._conditional_hidden == 0:
            tx = torch.cat([t, x], dim=1)
        else:
            tx = torch.cat([t, x,self.condition_vector], dim=1)
        return self._drift(tx)

    def g(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        t = t.expand(x.size(0), 1)
        if self._conditional_hidden == 0:
            tx = torch.cat([t, x], dim=1)
        else:
            tx = torch.cat([t, x,self.condition_vector], dim=1)
        return self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)

    def set_condition(self,encoder_forward):
        # encoder_forward has shape(batch_size, conditional_hidden)
        self.condition_vector = encoder_forward

###################
# Now we wrap it up into something that computes the SDE.
###################
class Generator(torch.nn.Module):
    def __init__(self, data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers,
                 conditional_hidden=0,conditional_init=False,conditional=False):
        super(Generator, self).__init__()

        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size
        self._conditional_init = conditional_init
        self.conditional = conditional

        if conditional_init:
            if self.conditional:
                self._initial = MLP(initial_noise_size+conditional_hidden+data_size, hidden_size,
                                    mlp_size, num_layers, tanh=False)
            else:
                self._initial = MLP(initial_noise_size + conditional_hidden, hidden_size, mlp_size, num_layers,
                                    tanh=False)
        else:
            self._initial = MLP(initial_noise_size, hidden_size, mlp_size, num_layers, tanh=False)

        self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers,conditional_hidden)
        self._readout = torch.nn.Linear(hidden_size, data_size)

    def forward(self,series,ts,encoder_forward=None):
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.
        batch_size = series.shape[0]
        ###################
        # Set conditional
        ###################
        if encoder_forward is not None:
            self._func.set_condition(encoder_forward)

        if self._conditional_init:
            init_noise = torch.randn(batch_size, self._initial_noise_size, device=ts.device)
            if self.conditional:
                init_noise = torch.cat([init_noise, encoder_forward,series[:, 0, :]], dim=1)
            else:
                init_noise = torch.cat([init_noise, encoder_forward], dim=1)

        ###################
        # Actually solve the SDE.
        ###################
        x0 = self._initial(init_noise)
        xs = torchsde.sdeint(self._func, x0, ts, method='midpoint', dt=1.0)  # shape (t_size, batch_size, hidden_size)
        xs = xs.transpose(0, 1)  # switch t_size and batch_size
        ys = self._readout(xs)

        ###################
        # Normalise the data to the form that the discriminator expects, in particular including time as a channel.
        ###################
        t_size = ts.size(0)
        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, t_size, 1)
        return torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2))


###################
# Next the discriminator. Here, we're going to use a neural controlled differential equation (neural CDE) as the
# discriminator, just as in the "Neural SDEs as Infinite-Dimensional GANs" paper. (You could use other things as well,
# but this is a natural choice.)
#
# There's actually a few different (roughly equivalent) ways of making the discriminator work. The curious reader is
# encouraged to have a read of the comment at the bottom of this file for an in-depth explanation.
###################
class DiscriminatorFunc(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers,conditional_hidden=0):
        super(DiscriminatorFunc, self).__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size
        self._conditional_hidden = conditional_hidden
        # tanh is important for model performance
        self._module = MLP(1 + hidden_size+conditional_hidden, hidden_size * (1 + data_size), mlp_size, num_layers, tanh=True)

    def forward(self, t, h):
        # t has shape ()
        # h has shape (batch_size, hidden_size)
        t = t.expand(h.size(0), 1)
        if self._conditional_hidden == 0:
            th = torch.cat([t, h], dim=1)
        else:
            th = torch.cat([t, h,self.condition_vector], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)

    def set_condition(self,encoder_forward):
        # encoder_forward has shape(batch_size, conditional_hidden)
        self.condition_vector = encoder_forward

class Discriminator(torch.nn.Module):

    def __init__(self, data_size, hidden_size, mlp_size, num_layers,conditional_hidden=0,conditional_init=True):
        super(Discriminator, self).__init__()

        if conditional_init:
            self._initial = self._initial = MLP(1 + data_size+conditional_hidden, hidden_size, mlp_size, num_layers, tanh=False)
        else:
            self._initial = self._initial = MLP(1 + data_size, hidden_size, mlp_size, num_layers, tanh=False)

        self._conditional_init = conditional_init
        self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers,conditional_hidden)
        self._readout = torch.nn.Linear(hidden_size, 1)

    def forward(self,ys_coeffs,encoder_forward=None):
        # ys_coeffs has shape (batch_size, t_size, 1 + data_size)
        # The +1 corresponds to time. When solving CDEs, It turns out to be most natural to treat time as just another
        # channel: in particular this makes handling irregular data quite easy, when the times may be different between
        # different samples in the batch.
        ###################
        # Set conditional
        ###################
        Y = torchcde.LinearInterpolation(ys_coeffs)
        Y0 = Y.evaluate(Y.interval[0])

        if encoder_forward is not None:
            self._func.set_condition(encoder_forward)
            if self._conditional_init:
                Y0 = torch.cat([Y0, encoder_forward], dim=1)

        h0 = self._initial(Y0)
        hs = torchcde.cdeint(Y, self._func, h0, Y.interval, adjoint=False, method='midpoint',
                             options=dict(step_size=1.0))  # shape (batch_size, 2, hidden_size)
        score = self._readout(hs[:, -1])
        return score.mean()


def train_generator(ts, batch_size, generator, discriminator, generator_optimiser, discriminator_optimiser):
    generated_samples = generator(ts, batch_size)
    generated_score = discriminator(generated_samples)

    generated_score.backward()
    generator_optimiser.step()
    generator_optimiser.zero_grad()
    discriminator_optimiser.zero_grad()


def train_discriminator(ts, batch_size, real_samples, generator, discriminator, discriminator_optimiser, gp_coeff):
    with torch.no_grad():
        generated_samples = generator(ts, batch_size)
    generated_score = discriminator(generated_samples)

    real_score = discriminator(real_samples)

    penalty = gradient_penalty(generated_samples, real_samples, discriminator)
    loss = generated_score - real_score
    (gp_coeff * penalty - loss).backward()
    discriminator_optimiser.step()
    discriminator_optimiser.zero_grad()


def evaluate_loss(ts, batch_size, dataloader, generator, discriminator):
    with torch.no_grad():
        total_samples = 0
        total_loss = 0
        for real_samples, in dataloader:
            generated_samples = generator(ts, batch_size)
            generated_score = discriminator(generated_samples)

            real_score = discriminator(real_samples)

            loss = generated_score - real_score

            batch_size = real_samples.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
    return total_loss / total_samples