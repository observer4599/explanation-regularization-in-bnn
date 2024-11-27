import torch
import torch.nn.functional as F
from typing import Union, Tuple, Optional
from torch.nn.init import kaiming_uniform_, _calculate_fan_in_and_fan_out, uniform_, constant_
import math

EPS: float = 1e-8


class BayesianConvolution(torch.nn.Module):
    def __init__(self, rho: float, in_channels, out_channels, kernel_size, stride=1,
                 padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.return_full = True

        # Weight parameters
        # mu
        self.weight = torch.nn.Parameter(kaiming_uniform_(torch.empty((self.out_channels, self.in_channels,
                                                                       self.kernel_size, self.kernel_size)),
                                                          nonlinearity='relu'))

        # rho
        self.weight_rho = torch.nn.Parameter(constant_(torch.empty((self.out_channels, self.in_channels,
                                                                    self.kernel_size, self.kernel_size)),
                                                       rho))
        # Bias parameters
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        # mu
        self.bias_mu = torch.nn.Parameter(
            uniform_(torch.empty(self.out_channels), -bound, bound))
        # rho
        self.bias_rho = torch.nn.Parameter(
            constant_(torch.empty(self.out_channels), rho))

    def forward(self, input_, sample: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return forward(self, input_, sample)

    def kl_loss(self) -> float:
        return kl_loss(self)

    @property
    def weight_mean(self):
        return self.weight

    @property
    def bias_mean(self):
        return self.bias_mu

    @property
    def weight_sigma(self):
        return F.softplus(self.weight_rho) + EPS

    @property
    def bias_sigma(self):
        return F.softplus(self.bias_rho) + EPS


class BayesianLinear(torch.nn.Module):
    def __init__(self, rho: float, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.return_full = True

        # Weight parameters
        # mu
        self.weight = torch.nn.Parameter(kaiming_uniform_(torch.empty(output_size, input_size),
                                                          nonlinearity='relu'))
        # rho_pi
        self.weight_rho = torch.nn.Parameter(
            constant_(torch.empty(output_size, input_size), rho))

        # Bias parameters
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        # mu
        self.bias_mu = torch.nn.Parameter(
            uniform_(torch.empty(output_size), -bound, bound))
        # rho
        self.bias_rho = torch.nn.Parameter(
            constant_(torch.empty(output_size), rho))

    def forward(self, input_, sample: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return forward(self, input_, sample)

    def kl_loss(self) -> float:
        return kl_loss(self)

    @property
    def weight_mean(self):
        return self.weight

    @property
    def bias_mean(self):
        return self.bias_mu

    @property
    def weight_sigma(self):
        return F.softplus(self.weight_rho) + EPS

    @property
    def bias_sigma(self):
        return F.softplus(self.bias_rho) + EPS


def forward(layer: Union[BayesianLinear, BayesianConvolution], x: torch.Tensor,
            sample: bool):
    if isinstance(layer, BayesianConvolution):
        act_mu = F.conv2d(x, layer.weight_mean, layer.bias_mean,
                          layer.stride, layer.padding)

        act_var = F.conv2d(torch.square(x), torch.square(layer.weight_sigma), torch.square(layer.bias_sigma),
                           layer.stride, layer.padding)
    else:
        act_mu = F.linear(x, layer.weight_mean, layer.bias_mean)
        act_var = F.linear(torch.square(x), torch.square(
            layer.weight_sigma), torch.square(layer.bias_sigma))

    act_std = torch.sqrt(act_var + EPS)
    if layer.training or sample:
        if layer.return_full:
            return act_mu + act_std * torch.randn(act_mu.shape, device=act_mu.device), act_mu, act_std
        else:
            return act_mu + act_std * torch.randn(act_mu.shape, device=act_mu.device)
    if layer.return_full:
        return act_mu, act_mu, act_std
    return act_mu


def kl_loss(layer: Union[BayesianLinear, BayesianConvolution]) -> float:
    weight = calculate_kl(layer.weight_mean, layer.weight_sigma)
    bias = calculate_kl(layer.bias_mean, layer.bias_sigma)
    return weight + bias


def calculate_kl(mu_q: torch.Tensor, sig_q: torch.Tensor) -> float:
    """
    Kingma, D. P. & Welling, M. Auto-Encoding Variational Bayes. arXiv [stat.ML] (2013)
    """
    mu_p = torch.tensor([0], device=mu_q.device)
    sig_p = torch.tensor([.1], device=mu_q.device)
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 +
                (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2))
    return kl.sum()
