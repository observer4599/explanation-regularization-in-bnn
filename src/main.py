import joblib
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import tyro
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from layers import BayesianLinear, BayesianConvolution
from torch.distributions.normal import Normal
from utils import get_project_root
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import Accuracy, AUROC, F1Score
from captum.attr import Saliency
from torch.optim.lr_scheduler import ExponentialLR
from lightning.pytorch.callbacks import LearningRateMonitor
from bnn import BayesianNeuralNetwork
from data_utils import MEAN, STD


class LitBNN(L.LightningModule):
    def __init__(self, dataset: str, feedback_weight: float,
                 num_mc_samples: int, lr: float, feedback: bool, train_num_mb: int,
                 val_num_mb: int, rho: float, anneal_lr_rate: float, data_augmentation: bool) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.data_augmentation = data_augmentation
        if "decoy_mnist" in self.dataset:
            self.out_dim = 10
        elif "isic" in self.dataset:
            self.out_dim = 2
        elif "oxford-iiit-pet" in self.dataset:
            self.out_dim = 37
        elif "sbd" in self.dataset:
            self.out_dim = 5
        else:
            raise ValueError()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.out_dim, average="macro")
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.out_dim, average="macro")
        self.train_auc = AUROC(task="multiclass", num_classes=self.out_dim)
        self.val_auc = AUROC(task="multiclass", num_classes=self.out_dim)
        self.train_f1 = F1Score(task="multiclass", num_classes=self.out_dim)
        self.val_f1 = F1Score(task="multiclass", num_classes=self.out_dim)

        self.feedback_weight = feedback_weight
        self.num_mc_samples = num_mc_samples
        self.lr = lr
        self.feedback = feedback
        self.train_num_mb = train_num_mb
        self.val_num_mb = val_num_mb
        self.anneal_lr_rate = anneal_lr_rate

        self.bnn = BayesianNeuralNetwork(dataset, rho)
        self.hook_layer()
        self.saliency = Saliency(self.bnn)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y, mask = batch
        if self.data_augmentation:
            # https://www.nature.com/articles/s42256-020-0212-3
            with torch.no_grad():
                norm_mean = torch.tensor(MEAN, device=self.device).unsqueeze(1).unsqueeze(2).unsqueeze(0)
                norm_std = torch.tensor(STD, device=self.device).unsqueeze(1).unsqueeze(2).unsqueeze(0)
                mask_bc = mask.repeat((1, x.shape[1], 1, 1))
                x[mask_bc == 1] = 0
                noise = torch.rand_like(mask_bc.float())
                if noise.shape[1] == 3:
                    noise = (noise - norm_mean) / norm_std
                # Added the check for the last experiment (ISIC, data_augmentation) due to some bug that I cannot find
                if x[mask_bc == 1].shape == noise[mask_bc == 1].shape:
                    x[mask_bc == 1] = noise[mask_bc == 1]
                else:
                    print("An irregularity happened!")

        nll, kl, exp_nll, y_hat = self.loss(x, y, mask, self.train_num_mb, True)
        self.train_accuracy(y_hat, y)
        self.train_auc(y_hat, y)
        self.train_f1(y_hat, y)

        loss = nll + kl
        log_dict = {"train/loss": loss, "train/nll": nll, "train/kl": kl, "train/acc_step": self.train_accuracy,
                    "train/auc": self.train_auc, "train/f1": self.train_f1}

        if self.feedback:
            loss += self.feedback_weight * exp_nll
            log_dict.update({"train/exp_nll": exp_nll})
        self.log_dict(log_dict, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        nll, kl, exp_nll, y_hat = self.loss(x, y, mask, self.val_num_mb, False)
        self.val_accuracy(y_hat, y)
        self.val_f1(y_hat, y)
        self.val_auc(y_hat, y)

        # Saliency
        saliency = self.saliency.attribute(x, target=y)
        saliency_cover = saliency[torch.nonzero(mask, as_tuple=True)].sum() / saliency.sum()

        loss = nll + kl
        log_dict = {"val/loss": loss, "val/nll": nll, "val/kl": kl, "val/acc": self.val_accuracy,
                    "val/saliency": saliency_cover, "val/auc": self.val_auc, "val/f1": self.val_f1}

        if self.feedback:
            loss += self.feedback_weight * exp_nll
            log_dict.update({"val/exp_nll": exp_nll})
        self.log_dict(log_dict, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ExponentialLR(optimizer, self.anneal_lr_rate),
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def calculate_kl(self):
        """Return the summed log prior probabilities from all layers."""
        parameters = sum([module.kl_loss() for module in list(self.bnn.features) + list(self.bnn.classifier)
                          if isinstance(module, (BayesianConvolution, BayesianLinear))])
        return parameters

    def hook_layer(self):
        def hook_function(module, x, output):
            _, self.act, self.act_std = output

        # Register hook to the first layer
        self.bnn.target_layer.register_forward_hook(hook_function)

    def loss(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, num_mb: int, train: bool):
        """Sample the elbo, implementing the minibatch version presented in section 3.4 in BBB paper."""
        outputs = torch.zeros(self.num_mc_samples, x.shape[0], self.out_dim, device=self.device)
        # Feedback cost
        exp_nll = torch.zeros(self.num_mc_samples, device=self.device)

        if self.feedback:
            if "decoy_mnist" in self.dataset:
                mask = F.adaptive_max_pool2d(mask, (11, 11))
                mask = mask.repeat(1, 16, 1, 1)
            elif "isic" in self.dataset:
                mask = F.adaptive_max_pool2d(mask, (13, 13))
                mask = mask.repeat(1, 256, 1, 1)
            elif "oxford-iiit-pet" in self.dataset or "sbd" in self.dataset:
                mask = 1 - F.adaptive_max_pool2d(1 - mask, (13, 13))
                mask = mask.repeat(1, 256, 1, 1)
            else:
                raise NotImplementedError()

            index = torch.nonzero(mask, as_tuple=True)
            for i in range(self.num_mc_samples):
                outputs[i] = self.bnn(x, True, mask=mask if train else None)

                exp_nll[i] = -Normal(self.act[index], self.act_std[index]).log_prob(
                    torch.zeros_like(self.act[index])).sum()
        else:
            for i in range(self.num_mc_samples):
                outputs[i] = self.bnn(x, True, mask=None)

        # Complexity cost
        kl = self.calculate_kl() / num_mb

        # Data cost
        y_hat = outputs.mean(0)

        if self.dataset == "isic":
            if not hasattr(self, "class_weight"):
                self.class_weight = torch.tensor([1.0, 11618 / 1376], device=self.device)
            nll = F.cross_entropy(
                input=y_hat, target=y, reduction='sum', weight=self.class_weight
            )
        else:
            nll = F.cross_entropy(input=y_hat, target=y, reduction='sum')

        return nll, kl, exp_nll.mean(), y_hat


@dataclass
class Args:
    feedback: bool = False
    data_augmentation: bool = True
    dataset: str = "isic"
    accelerator: str = "gpu"
    # All: gpu, Decoy MNIST: cpu
    num_mc_samples: int = 1
    # All: 1
    lr: float = 1e-4
    # Decoy MNIST: 1e-3, ISIC: 1e-4
    feedback_weight: float = 1e-5
    # Decoy MNIST Color: 5e-5, Decoy MNIST Position; 3e-4, ISIC: 1e-5, Oxford-IIIT-Pet: 2e-6, SBD: 2e-6
    num_epochs: int = 60
    # Decoy MNIST: 200, ISIC: 60, Oxford-IIIT-Pet: 100, SBD: 100
    seed: int = 8434
    # https://www.random.org/, Min: 0, Max: 10000, 2024-10-16 07:49:07 UTC
    num_workers: int = 8
    # All: 8
    batch_size: int = 128
    # All: 128
    check_val_every_n_epoch: int = 3
    # All: 5, ISIC: 3
    rho: float = -3.8
    # Decoy MNIST: -2, ISIC: -3.8, Oxford-IIIT-Pet: -4.2, SBD: -3.8
    anneal_lr_rate: float = 0.98
    # All: 0.98, Decoy MNIST: 0.99
    val_idx: int = 2
    num_split: int = 3
    # All: 3
    remove_exp: float = 0.3
    # All: 0.3


def main():
    args = tyro.cli(Args)
    L.seed_everything(args.seed, workers=True, verbose=True)
    assert args.dataset in ("decoy_mnist_color", "decoy_mnist_position", "isic",
                            "oxford-iiit-pet", "sbd")
    assert not (args.feedback and args.data_augmentation)
    assert args.num_split > args.val_idx and args.val_idx >= 0 and int(args.val_idx) == args.val_idx, \
        "Chosen split is not valid"

    if args.feedback:
        mode = "feedback"
    elif args.data_augmentation:
        mode = "data_augmentation"
    else:
        mode = "none"

    logger = TensorBoardLogger(get_project_root(), f"runs/{args.dataset}/{mode}")
    logger.log_hyperparams(vars(args))
    data_folder = get_project_root() / "data" / args.dataset

    # Load data
    with (data_folder / "trainval.joblib").open("rb") as f:
        x_trainval, y_trainval, mask_trainval = joblib.load(f)


    # Trainval split
    data_idx = np.arange(x_trainval.shape[0])
    np.random.shuffle(data_idx)
    data_idx = np.array_split(data_idx, args.num_split)
    train_idx = []
    for i in range(len(data_idx)):
        if i == args.val_idx:
            val_idx = data_idx[i]
        else:
            train_idx.append(data_idx[i])
    train_idx = np.concatenate(train_idx)
    assert len(x_trainval) == (len(train_idx) + len(val_idx))

    # Training data
    x_train, y_train, mask_train = x_trainval[train_idx], y_trainval[train_idx], mask_trainval[train_idx]
    # Do not regularize the entire training dataset.
    # Write that we cannot handle decoy_mnist_position without doing this.
    idx = np.arange(0, len(mask_train), 1)
    np.random.shuffle(idx)
    mask_train[idx[:int(len(idx) * args.remove_exp)]] = 0

    train_loader = DataLoader(TensorDataset(x_train, y_train.long(), mask_train),
                              num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True,
                              persistent_workers=True)
    args.train_num_mb = len(train_loader)

    # Validation data
    x_val, y_val, mask_val = x_trainval[val_idx], y_trainval[val_idx], mask_trainval[val_idx]
    val_loader = DataLoader(TensorDataset(x_val, y_val.long(), mask_val),
                            num_workers=args.num_workers, batch_size=args.batch_size, persistent_workers=True)
    args.val_num_mb = len(val_loader)

    bnn = LitBNN(args.dataset, args.feedback_weight, args.num_mc_samples, args.lr,
                 args.feedback, args.train_num_mb, args.val_num_mb, args.rho,
                 args.anneal_lr_rate, args.data_augmentation)

    if "isic" in args.dataset or "oxford-iiit-pet" in args.dataset or "sbd" in args.dataset:
        alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet',
                                 weights="AlexNet_Weights.IMAGENET1K_V1").state_dict()
        del alexnet["classifier.6.weight"]
        del alexnet["classifier.6.bias"]
        bnn.bnn.load_state_dict(alexnet, strict=False)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    if "isic" in args.dataset:
        monitor = "val/auc"
    else:
        monitor = "val/acc"
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode="max", verbose=True,
                                          auto_insert_metric_name=False)
    trainer = L.Trainer(max_epochs=args.num_epochs, logger=logger, accelerator=args.accelerator,
                        check_val_every_n_epoch=args.check_val_every_n_epoch,
                        callbacks=[lr_monitor, checkpoint_callback])

    trainer.fit(model=bnn, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
