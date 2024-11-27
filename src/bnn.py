from torch import nn
from layers import BayesianLinear, BayesianConvolution
import torch


class BayesianNeuralNetwork(nn.Module):
    def __init__(self, dataset: str, rho: float) -> None:
        super(BayesianNeuralNetwork, self).__init__()
        self.rho = rho
        self.return_full = True

        if "decoy_mnist" in dataset:
            self.features, self.classifier = self.construct_lenet(10)
            self.target_layer = self.features[3]
        elif "isic" in dataset:
            self.features, self.classifier = self.construct_alexnet(2)
            self.target_layer = self.features[10]
        elif "oxford-iiit-pet" in dataset:
            self.features, self.classifier = self.construct_alexnet(37)
            self.target_layer = self.features[10]
        elif "sbd" in dataset:
            self.features, self.classifier = self.construct_alexnet(5)
            self.target_layer = self.features[10]
        else:
            raise ValueError()

    def construct_lenet(self, out_dim):
        # https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html?highlight=lenet
        features = nn.Sequential(
            BayesianConvolution(self.rho, 1, 6, 3),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            BayesianConvolution(self.rho, 6, 16, 3),
            nn.ReLU(), nn.MaxPool2d(2, 2))
        classifier = nn.Sequential(
            BayesianLinear(self.rho, 16 * 5 * 5, 120), nn.ReLU(),
            BayesianLinear(self.rho, 120, 84), nn.ReLU(),
            BayesianLinear(self.rho, 84, out_dim)
        )
        return features, classifier

    def construct_alexnet(self, out_dim, dropout: float = 0.5):
        features = nn.Sequential(
            BayesianConvolution(self.rho, 3, 64, 11, 4, 2),
            nn.ReLU(), nn.MaxPool2d(3, 2),
            BayesianConvolution(self.rho, 64, 192, 5, padding=2),
            nn.ReLU(), nn.MaxPool2d(3, 2),
            BayesianConvolution(self.rho, 192, 384, 3, padding=1),
            nn.ReLU(),
            BayesianConvolution(self.rho, 384, 256, 3, padding=1),
            nn.ReLU(),
            BayesianConvolution(self.rho, 256, 256, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(3, 2),
        )

        classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            BayesianLinear(self.rho, 256 * 6 * 6, 4096),
            nn.ReLU(), nn.Dropout(p=dropout),
            BayesianLinear(self.rho, 4096, 4096),
            nn.ReLU(),
            BayesianLinear(self.rho, 4096, out_dim)
        )
        return features, classifier

    def forward(self, x: torch.Tensor, sample: bool = False, mask=None):
        """Propagate the input through the network."""
        for module in self.features:
            if isinstance(module, BayesianConvolution):
                x = module(x, sample)[0] if self.return_full else module(x, sample)
                if mask is not None and module is self.target_layer:
                    x[mask == 1.0] = 0
            else:
                x = module(x)

        x = torch.flatten(x, 1)

        for module in self.classifier:
            if isinstance(module, BayesianLinear):
                x = module(x, sample)[0] if self.return_full else module(x, sample)
            else:
                x = module(x)
        return x

    def set_return_full(self, full: bool):
        self.return_full = full
        for module in self.features:
            if isinstance(module, BayesianConvolution):
                module.return_full = False
        for module in self.classifier:
            if isinstance(module, BayesianLinear):
                module.return_full = False
