from torch import  nn
import torch


class Exp(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Sigmoid.png

    Examples::

        >>> m = nn.Exp()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.exp(input)


class MLP(nn.Module):

    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.input_dim = kwargs.get("input_dim")
        self.output_dim = kwargs.get("output_dim")
        self.layers_dim = kwargs.get("layers_dim")
        self.dropout = kwargs.get("dropout", .4)
        self.normalization = kwargs.get("normalization")

        self.layers_dim = [self.input_dim] + self.layers_dim + [self.output_dim]
        self.num_layer = len(self.layers_dim)
        self.output_transformation = kwargs.get("ouput_transformation")
        self.define_deep_parameters()

    def sample_parameters(self):
        parameters = {"input_dim": 2, "layers_dim": [3, 2], "ouput_dim": 3, "normalization": True, "ouput_transformation": None}
        return parameters

    def forward(self, x):
        return self.perceptron(x)

    def define_deep_parameters(self):
        self.perceptron = nn.ModuleList([])
        for layer_index in range(self.num_layer - 1):
            self.perceptron.append(nn.Linear(self.layers_dim[layer_index], self.layers_dim[layer_index + 1]))
            if self.dropout > 0 and layer_index != self.num_layer - 2:
                self.perceptron.append(nn.Dropout(self.dropout))
            # if self.normalization and layer_index < self.num_layer - 2:
            #     self.perceptron.append(nn.BatchNorm1d(self.layers_dim[layer_index  1]))
            if layer_index != self.num_layer - 2:
                if layer_index < self.num_layer - 1 and self.num_layer > 2:
                    self.perceptron.append(nn.ReLU())
        if self.output_transformation == "relu":
            self.perceptron.append(nn.ReLU())
        elif self.output_transformation == "sigmoid":
            self.perceptron.append(nn.Sigmoid())
        elif self.output_transformation == "exp":
            self.perceptron.append(Exp())

        self.perceptron = nn.Sequential(*self.perceptron)

    def init_parameters(self):
        for layer in self.perceptron:
            if hasattr(layer, 'weight'):
                if isinstance(layer, (nn.InstanceNorm2d, nn.LayerNorm)):
                    nn.init.normal_(layer.weight, mean=1., std=0.02)
                else:
                    nn.init.xavier_normal_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias, 0.)


class CNNBlock(nn.Module):
    """
    Block of Convolutional neural networks
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, nonlinearity, dim, normalization=True, auxiliary_noise=False):

        super(CNNBlock, self).__init__()

        layers = nn.ModuleList([])
        n_layers = len(out_channels)
        for i in range(n_layers):
            ch_in = in_channels if i == 0 else out_channels[i - 1]

            # cnn:
            print("in_ch =", ch_in, "out_ch =", out_channels[i], "k =", kernel_size[i], "p =", padding[i], "s =", stride[i], "(H, W) =", dim[i])
            layers.append(nn.Conv2d(ch_in, out_channels[i], kernel_size[i], padding=padding[i], stride=stride[i]))

            # layer/instance normalization:
            if normalization:
                layers.append(nn.InstanceNorm2d(out_channels[i], affine=True))

            # non-linearity:
            layers.append(nonlinearity())

        self.layers = nn.Sequential(*layers)
        self.param_init()

    def param_init(self):
        """
        Parameters initialization.
        """
        for layer in self.modules():
            if hasattr(layer, 'weight'):
                if isinstance(layer, (nn.InstanceNorm2d, nn.LayerNorm)):
                    nn.init.normal_(layer.weight, mean=1., std=0.02)
                else:
                    nn.init.xavier_normal_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias, 0.)

    def forward(self, x):
        return self.layers(x)
