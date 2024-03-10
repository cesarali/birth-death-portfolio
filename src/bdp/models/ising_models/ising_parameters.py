import os
import sys
import tqdm
import time
import torch
import random

import numpy as np
from torch import nn
from pprint import pprint
from torch.optim import Adam
from torch import matmul as m
from matplotlib import pyplot as plt
from torch.distributions import Normal, Bernoulli, Exponential

#from discrete_diffusion.models_.ising_models import
#from deep_fields.models_.generative_models.diffusion_probabilistic.discrete.ising_utils import obtain_all_spin_states, obtain_new_spin_states
#from deep_fields.models_.generative_models.diffusion_probabilistic.utils import get_timestep_embedding
#from deep_fields.models_.utils.basic_setups import create_dir_and_writer


class bernoulli_spins():

    def __init__(self,p):
        self.p = p
        self.bernoulli_distribution = Bernoulli(p)

    def sample(self,sample_shape):
        sample_one_zeros = self.bernoulli_distribution.sample(sample_shape=sample_shape)
        sample_one_zeros[torch.where(sample_one_zeros == 0)] = -1.
        return sample_one_zeros

def obtain_new_states(counted_states,flip_mask):
    number_of_counted_states,number_of_spins = counted_states.shape
    repeated_counted_states = torch.repeat_interleave(counted_states,number_of_spins,dim=0)
    repeated_mask = torch.tile(flip_mask,(number_of_counted_states,1))
    new_states = repeated_counted_states*repeated_mask
    return new_states

# hamiltonian
def Hamiltonian(J,x):
    H = m(x,J[None,:,:]).squeeze()
    H = torch.einsum('bi,bi->b',H,x)
    return H

# neighbors energy change
def Hamiltonian_i(J,x,i_random):
    J_i = J[:,i_random].T
    H_i = J[i_random,i_random] + torch.einsum('bi,bi->b',J_i,x)
    return H_i

# symmetric function
def symetric_function(J,new_states,energy=False):
    """
    FIX FOR DIAGONAL STATES, MAKE J ZERO
    """
    J_i = torch.diagonal(J)
    H_i = torch.einsum('sbi,sbi->sb',J[None,:,:],new_states)
    H_i = J_i + H_i
    f_s = mu/(2.*torch.cosh(H_i))
    if energy:
        return f_s.reshape(-1)
    else:
        return f_s.sum(axis=-1)

def initialize_model(number_of_spins,number_of_paths,J_mean,J_std):
    # define random interactions (fields in the diagonal)
    J = Normal(J_mean,J_std)
    J = J.sample(sample_shape=(number_of_spins,number_of_spins))
    J = (J + J.T)*.5

    # define uniform spin configurations
    x = torch.randint(0,2,(number_of_paths,number_of_spins))
    x[torch.where(x==0)] = -1
    x = x.float()
    paths = x.unsqueeze(1)

    return paths, J

def glauber_dynamics(T, tau, paths,J):
    number_of_paths = paths.shape[0]
    number_of_spins = J.shape[0]

    rows_index = torch.arange(0, number_of_paths)
    time_grid = torch.arange(0., T, tau)

    for t_i in time_grid:
        x = paths[:, -1, :]

        i_random = torch.randint(0, number_of_spins, (number_of_paths,))

        H_i = Hamiltonian_i(J, x, i_random)
        x_i = torch.diag(x[:, i_random])
        flip_probability = tau * mu * torch.exp(-x_i * H_i) / 2 * torch.cosh(H_i)
        r = torch.rand((number_of_paths,))
        where_to_flip = r < flip_probability
        x[(rows_index, i_random)][torch.where(where_to_flip)] = x[(rows_index, i_random)][
                                                                    torch.where(where_to_flip)] * -1.
        paths = torch.cat([paths, x.unsqueeze(1)], dim=1)

    return paths

def parametric_dynamics(paths,time_grid,number_of_paths,number_of_spins,time_embedding_dim,PHI_1):
    """
    so we pretty much extend the poisson in the grid, but instead of sampling
    from the number of arrivals wejust check if it arrived at the gap

    :return:
    """
    times = torch.zeros(number_of_paths).unsqueeze(1)
    paths = initial_distribution.sample([number_of_paths]).unsqueeze(1)
    # Obtain Transitions
    time_step = 0
    for time_index in time_grid[1:]:
        # embbedd time
        times_embedded = get_timestep_embedding(times[:, -1], time_embedding_dim=time_embedding_dim)
        times_embedded_repeated = torch.repeat_interleave(times_embedded, number_of_spins, dim=0)

        x = paths[:, int(time_index), :]
        x_repeated = torch.repeat_interleave(x, number_of_spins, dim=0)
        x_n_time = torch.hstack([x_repeated, times_embedded_repeated])

        new_states = obtain_new_states(x)
        xnew_n_time = torch.hstack([new_states, times_embedded_repeated])

        # symetric function
        f_ss = symetric_function(J, new_states.reshape(number_of_paths, number_of_spins, number_of_spins))
        f_ss = torch.repeat_interleave(f_ss, number_of_spins, dim=0)

        # parametric function
        phi_1_x = torch.sigmoid(PHI_1(x_n_time)) ** 2.
        phi_1_new = torch.sigmoid(PHI_1(xnew_n_time)) ** 2.
        rates = (phi_1_new / phi_1_x) * f_ss.unsqueeze(-1)
        rates = rates.reshape(number_of_paths, number_of_spins)
        transition_times = torch.min(Exponential(rates).sample(), dim=1)

        new_states = new_states.reshape(number_of_paths, number_of_spins, number_of_spins)
        new_states = torch.einsum("iis->is", new_states[:, transition_times.indices, :])

        where_keep_old = torch.where(transition_times.values > tau)
        new_states[where_keep_old] = x[where_keep_old]

        # push time forward
        times_now = times[:, -1]
        times_now = torch.full_like(times_now, time_index)

        paths = torch.cat([paths, new_states.unsqueeze(1)], dim=1)
        times = torch.hstack([times, times_now[:, None]])

    return paths,times

class ParametrizedIsingHamiltonian(nn.Module):
    """
    Simple Hamiltonian Model for Parameter Estimation
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------

        number_of_spins: int
        obtain_partition_function: bool
            only defined for number_of_spins < 10
        """
        super(ParametrizedIsingHamiltonian, self).__init__()
        self.beta = kwargs.get("beta")
        self.number_of_spins = kwargs.get("number_of_spins")

        self.lower_diagonal_indices = torch.tril_indices(self.number_of_spins, self.number_of_spins, -1)
        self.number_of_couplings = self.lower_diagonal_indices[0].shape[0]
        self.coupling_matrix = torch.zeros((self.number_of_spins,self.number_of_spins))
        self.flip_mask = torch.ones((self.number_of_spins, self.number_of_spins))
        self.flip_mask.as_strided([self.number_of_spins], [self.number_of_spins + 1]).copy_(torch.ones(self.number_of_spins) * -1.)
        self.model_identifier = str(int(time.time()))
        self.define_parameters(**kwargs)
        self.model_parameters = kwargs

    def define_parameters(self,**kwargs):
        self.couplings_sigma = kwargs.get("couplings_sigma")
        self.couplings_deterministic = kwargs.get("couplings_deterministic")
        # INITIALIZING COUPLINGS AND FIELDS
        if self.couplings_deterministic is None:
            if self.couplings_sigma is None:
                # we define the distributions in such a way that guarantee ergodicity of the glauber dynamics (std=1/number_of_spins)
                couplings = torch.clone(torch.Tensor(size=(
                    self.number_of_couplings,)).normal_(0., 1/float(self.number_of_spins)))
                fields = torch.clone(torch.Tensor(size=(
                    self.number_of_spins,)).normal_(0., 1/float(self.number_of_spins)))
            else:
                couplings = torch.clone(torch.Tensor(size=(
                    self.number_of_couplings,)).normal_(0., self.couplings_sigma))
                fields = torch.clone(torch.Tensor(size=(
                    self.number_of_spins,)).normal_(0., self.couplings_sigma))
        else:
            couplings = torch.ones(size=(self.number_of_couplings,))*self.couplings_deterministic
            fields = torch.ones(size=(self.number_of_spins,))*self.couplings_deterministic

        self.fields = nn.Parameter(fields)
        self.couplings = nn.Parameter(couplings)
        #========================================================================
        if kwargs.get("obtain_partition_function"):
            self.obtain_partition_function()

    def obtain_partition_function(self):
        all_states = obtain_all_spin_states(self.number_of_spins)
        with torch.no_grad():
            self.partition_function = self(all_states)
            self.partition_function = torch.exp(-self.partition_function).sum()

    def obtain_couplings_as_matrix(self,couplings=None):
        """
        converts the parameters vectors which store only the lower diagonal
        into a full symetric matrix

        :return:
        """
        if couplings is None:
            couplings = self.couplings
        coupling_matrix = torch.zeros((self.number_of_spins, self.number_of_spins))
        coupling_matrix[self.lower_diagonal_indices[0], self.lower_diagonal_indices[1]] = couplings
        coupling_matrix = coupling_matrix + coupling_matrix.T
        return coupling_matrix

    def log_probability(self,states):
        """
        :return:
        """
        return -self.beta*self(states).sum() + states.shape[0]*torch.log(self.partition_function)

    def sample(self,number_of_paths:int,number_of_mcmc_steps: int, **kwargs)-> torch.Tensor:
        """
        Here we follow a basic metropolis hasting algorithm

        :return:
        """
        self.number_of_paths = number_of_paths
        self.number_of_mcmc_steps = number_of_mcmc_steps

        # we start a simple bernoulli spin distribution
        p0 = torch.Tensor(self.number_of_spins * [0.5])
        initial_distribution = bernoulli_spins(p0)
        states = initial_distribution.sample(sample_shape=(self.number_of_paths,))
        paths = states.unsqueeze(1)
        rows_index = torch.arange(0, self.number_of_paths)

        # METROPOLIS HASTING
        for mcmc_index in range(self.number_of_mcmc_steps):
            i_random = torch.randint(0, self.number_of_spins, (self.number_of_paths,))
            index_to_change = (rows_index, i_random)

            states = paths[:,-1,:]
            new_states = torch.clone(states)
            new_states[index_to_change] = states[index_to_change] * -1.

            H_0 = self(states)
            H_1 = self(new_states)
            H = self.beta * (H_0-H_1)

            flip_probability = torch.exp(H)
            r = torch.rand((number_of_paths,))
            where_to_flip = r < flip_probability

            new_states = torch.clone(states)
            index_to_change = (rows_index[torch.where(where_to_flip)], i_random[torch.where(where_to_flip)])
            new_states[index_to_change] = states[index_to_change] * -1.

            paths = torch.cat([paths, new_states.unsqueeze(1)], dim=1)

        return paths

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Ising Hamiltonian

        Parameters
        ----------
        states:torch.Tensor
            Ising states defined by 1 or -1 spin values

        Returns
        -------
        Hamiltonian
        """
        coupling_matrix = self.obtain_couplings_as_matrix()
        H_couplings = torch.einsum('bi,bij,bj->b', states, coupling_matrix[None, :, :], states)
        H_fields = torch.einsum('bi,bi->b', self.fields[None, :], states)
        Hamiltonian = - H_couplings - H_fields
        return Hamiltonian

    @classmethod
    def get_parameters(self):
        kwargs = {
            "number_of_spins": 4,
            "beta": 1.,
            "obtain_partition_function": True,
            "couplings_deterministic":1.,
            "couplings_sigma":1.,
            "number_of_paths": 2,
            "number_of_mcmc_steps": 1000,
        }
        return kwargs

    def oppers_estimator(self, states: torch.Tensor) -> torch.Tensor:
        """
        Here we evaluate over the difference between the states and the corresponding
        one spin flop configuration

        Parameters
        ----------
        IsingHamiltonian:ParametrizedIsingHamiltonian

        states:torch.Tensor

        Returns
        -------
        loss
        """
        new_states = obtain_new_spin_states(states,self.flip_mask)

        H_states = self(states)
        H_new_states = self(new_states)
        H_new_states = H_new_states.reshape(H_states.shape[0], self.number_of_spins)
        H = self.beta * (H_new_states - H_states[:, None])
        H = torch.exp(-.5 * H)
        loss = H.sum()
        return loss

    def inference(self,mcmc_sample:torch.Tensor,real_fields:torch.Tensor,real_couplings:torch.Tensor,
                  number_of_epochs=10000,learning_rate=1e-3):
        writer, results_path, best_model_path = create_dir_and_writer(model_name="oppers_estimation",
                                                                      experiments_class="",
                                                                      model_identifier="ising_{0}".format(self.model_identifier),
                                                                      delete=True)
        states = mcmc_sample[:,-1,:]
        optimizer = Adam(self.parameters(), lr=learning_rate)
        norm_loss_history = []
        oppers_loss_history = []
        for i in tqdm.tqdm(range(number_of_epochs)):
            loss = self.oppers_estimator(states)
            if real_couplings is not None:
                couplings_norm = torch.norm(real_couplings - self.couplings)
                norm_loss_history.append(couplings_norm.item())
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            oppers_loss_history.append(loss.item())

            writer.add_scalar("train/loss", loss.item(), i)
            writer.add_scalar("train/norm", couplings_norm.item(), i)
            print("train loss {0}".format(loss.item()))

        print("Saving Model In {0}".format(best_model_path))
        torch.save(self, best_model_path)
        torch.save({"real_fields": real_fields,
                    "real_couplings": real_couplings,
                    "paths": mcmc_sample,
                    "oppers_loss_history":oppers_loss_history,
                    "norm_loss_history":norm_loss_history},
                   os.path.join(results_path, "real_couplings.tr"))
        print("Real Data in {0}".format(results_path))
        return best_model_path, results_path

#=============================================================
# ARGUMENTS
#=============================================================

if __name__=="__main__":

    #===========================================================================
    # TEST PARAMETRIC MODEL
    #===========================================================================
    kwargs = ParametrizedIsingHamiltonian.get_parameters()
    kwargs.update({"beta":1.,
                   "number_of_spins":3,
                   "couplings_deterministic":1.,
                   "couplings_sigma":1.})

    pprint(kwargs)

    PIH_real = ParametrizedIsingHamiltonian(**kwargs)
    mcmc_sample = PIH_real.sample(**kwargs)
    real_couplings = torch.clone(PIH_real.couplings)
    real_fields = torch.clone(PIH_real.fields)
    #===========================================================================
    # TEST MANFRED ESTIMATOR
    #===========================================================================
    #kwargs.update({"couplings_deterministic": None, "couplings_sigma": None})
    #PIH_train = ParametrizedIsingHamiltonian(**kwargs)
    #PIH_train.inference(mcmc_sample,real_fields,real_couplings,
    #                    number_of_epochs= 10000,learning_rate = 1e-3)
    #===========================================================================
    # TEST DYNAMICS
    #===========================================================================
    """
    #paths, J = initialize_model(number_of_spins,number_of_paths,J_mean,J_std)
    #paths = glauber_dynamics(T, tau, paths, J)

    time_step = 0
    depth_index = 0
    #x = paths[:, time_step, :]
    #x = lexicographical_ordering_of_spins(x)

    initial_distribution = bernoulli_spins(p0)
    final_distribution = bernoulli_spins(p1)

    PHI_0 = MLP(**phi_parameters)
    PHI_1 = MLP(**phi_parameters)

    time_grid = torch.arange(0., T, tau)
    paths = initial_distribution.sample([number_of_paths]).unsqueeze(1)
    paths,times = parametric_dynamics(paths, time_grid, number_of_paths, number_of_spins, time_embedding_dim, PHI_1)
    """