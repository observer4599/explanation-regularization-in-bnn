import shutil
import time
import numpy as np
from utils import get_project_root
from dataclasses import dataclass
import tyro
import lightning as L
import torch.nn.functional as F
from main import LitBNN
import joblib
from torch.utils.data import TensorDataset, DataLoader
import torch
from sklearn import metrics
from collections import defaultdict
import statistics
from tqdm import tqdm, trange
import seaborn as sns
from typing import Literal, Callable
from captum.attr import LayerGradCam, LayerAttribution, visualization, Saliency, DeepLift
from layers import BayesianLinear, BayesianConvolution
from collections import OrderedDict
import matplotlib.pyplot as plt
from data_utils import MEAN, STD
import scipy
import joblib

MEAN = np.array(MEAN)[..., np.newaxis, np.newaxis]
STD = np.array(STD)[..., np.newaxis, np.newaxis]


def load_model(path):
    model = LitBNN.load_from_checkpoint(path)
    model.eval()
    return model


def compute_balanced_accuracy(y, y_hat):
    return metrics.balanced_accuracy_score(y, np.argmax(y_hat, axis=1))


def compute_f1(y, y_hat):
    if y_hat.shape[1] == 2:
        return metrics.f1_score(y, np.argmax(y_hat, axis=1))
    return metrics.f1_score(y, np.argmax(y_hat, axis=1), average='macro')


def compute_roc_auc(y, y_hat):
    y_hat = scipy.special.softmax(y_hat, axis=1)
    if y_hat.shape[1] == 2:
        y_hat = y_hat[:, 1]
        return metrics.roc_auc_score(y, y_hat)
    return metrics.roc_auc_score(y, y_hat, multi_class="ovr")


def forward(self, x: torch.Tensor, sample: bool = False):
    """Propagate the input through the network."""
    for module in self.features:
        if isinstance(module, BayesianConvolution):
            x = module(x, sample)[0]
        else:
            x = module(x)

    x = torch.flatten(x, 1)

    for module in self.classifier:
        if isinstance(module, BayesianLinear):
            x = module(x, sample)[0]
        else:
            x = module(x)
    return x


def get_attrbutor(model, dataset):
    bnn = model.bnn
    bnn.set_return_full(False)
    # https://discuss.pytorch.org/t/how-do-i-remove-forward-hooks-on-a-module-without-the-hook-handles/140393/2
    bnn.target_layer._forward_hooks: dict[int, Callable] = OrderedDict()
    if "mnist" in dataset:
        layer_gc = Saliency(bnn)
    else:
        layer_gc = LayerGradCam(bnn, bnn.target_layer)
    return layer_gc


@dataclass
class Args:
    seed: int = 0
    dataset: str = "isic"
    mode: Literal["none", "feedback", "data_augmentation"] = "feedback"
    batch_size: int = 128
    n_samples: int = 1
    device: str = "mps"
    plot: bool = False
    data_type: Literal["trainval", "test"] = "test"
    with_patch: bool = True
    # Only use it for ISIC


def main():
    args = tyro.cli(Args)
    assert args.mode in ("feedback", "none", "data_augmentation")
    L.seed_everything(args.seed, True)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    device = torch.device(device)

    folder = get_project_root() / f"runs/{args.dataset}/{args.mode}"
    models = []
    fig_folder = []
    for run in folder.iterdir():
        if ".DS_Store" in str(run):
            continue
        fig_folder.append(run / "figures")
        model_checkpoint = list((run / "checkpoints").iterdir())[0]
        models.append(load_model(model_checkpoint).to(device))

    if args.plot:
        fig_folder = fig_folder[0] / args.data_type
        if fig_folder.exists():
            shutil.rmtree(fig_folder)
        fig_folder.mkdir(parents=True)

    # Load data
    with (get_project_root() / f"data/{args.dataset}/{args.data_type}.joblib").open("rb") as f:
        x_dataset, y_dataset, mask_dataset = joblib.load(f)
    dataloader = DataLoader(TensorDataset(x_dataset, y_dataset.long(), mask_dataset), batch_size=args.batch_size)

    result = defaultdict(list)

    if args.plot:
        bnn = models[0].bnn
        layer_gc = get_attrbutor(models[0], args.dataset)

        for batch in dataloader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            result["has_mask"].append((torch.sum(batch[2], dim=(1, 2, 3)) > 0).numpy(force=True))
            result["y"].append(y.numpy(force=True))
            result["y_hat"].append(F.softmax(bnn(x), dim=1).numpy(force=True))
            if "mnist" not in args.dataset:
                attr = layer_gc.attribute(x, target=y, relu_attributions=True)
                attr = LayerAttribution.interpolate(attr, (x.shape[2], x.shape[3]),
                                                    "bilinear")
            else:
                attr = layer_gc.attribute(x, target=y)
            result["attr"].append(np.transpose(attr.numpy(force=True), (0, 2, 3, 1)))
            img = x.numpy(force=True)
            if "mnist" not in args.dataset:
                img = img * STD + MEAN
            result["img"].append(np.transpose(img, (0, 2, 3, 1)))

        result["img"] = [np.clip(np.concatenate(result["img"], axis=0), 0, 1)]
        result["attr"] = [np.concatenate(result["attr"], axis=0)]
        result["y"] = [np.concatenate(result["y"])]
        result["y_hat"] = [np.concatenate(result["y_hat"])]
        result["has_mask"] = [np.concatenate(result["has_mask"])]

        for i in trange(0, min(len(result["img"][0]), 1_000)):
            for method in ("blended_heat_map", "original_image"):
                fig, ax = plt.subplots(constrained_layout=True)
                if method == "original_image" and "mnist" in args.dataset:
                    ax.imshow(result["img"][0][i], cmap="gray")
                    ax.axis("off")
                else:
                    visualization.visualize_image_attr(result["attr"][0][i] + 1e-8, result["img"][0][i],
                                                       plt_fig_axis=(fig, ax), use_pyplot=False, method=method,
                                                       cmap=sns.color_palette("coolwarm", as_cmap=True),
                                                       alpha_overlay=0.7)

                fig.savefig(
                    fig_folder / f"id={i}__method={method}__y={result['y'][0][i]}__y-hat={result['y_hat'][0][i] if args.dataset != 'oxford-iiit-pet' else None}.pdf",
                    bbox_inches='tight', pad_inches=0.0)
                fig.savefig(
                    fig_folder / f"id={i}__method={method}__y={result['y'][0][i]}__y-hat={result['y_hat'][0][i] if args.dataset != 'oxford-iiit-pet' else None}.svg",
                    bbox_inches='tight', pad_inches=0.0)
                joblib.dump((result["img"][0][i], result["attr"][0][i] + 1e-8, result["has_mask"][0][i]), fig_folder / \
                            f"id={i}__method={method}__y={result['y'][0][i]}__y-hat={result['y_hat'][0][i] if args.dataset != 'oxford-iiit-pet' else None}.joblib")
                plt.close(fig)

    L.seed_everything(args.seed, True)
    result = defaultdict(list)
    for model in tqdm(models):
        layer_gc = get_attrbutor(model, args.dataset)
        y, y_hat, overlaps = [], [], []
        for _ in range(args.n_samples):
            for batch in dataloader:
                x = batch[0].to(device)
                labels = batch[1].to(device)
                mask = batch[2].to(device)
                if not args.with_patch and "isic" in args.dataset:
                    no_patch = torch.sum(mask, dim=(1, 2, 3)) == 0
                    x, labels, mask = x[no_patch], labels[no_patch], mask[no_patch]
                y.append(labels.numpy(force=True))
                y_hat.append(model.bnn(x, sample=True).numpy(force=True))

                if "mnist" not in args.dataset:
                    attr = layer_gc.attribute(x, target=labels, relu_attributions=True)
                    attr = LayerAttribution.interpolate(attr, (x.shape[2], x.shape[3]),
                                                        "bilinear")
                else:
                    attr = layer_gc.attribute(x, target=labels)
                overlap = torch.sum(mask * attr, dim=(1, 2, 3)) / (torch.sum(attr, dim=(1, 2, 3)) + 1e-8)
                overlaps.append(overlap.numpy(force=True))

        y, y_hat, overlaps = np.concatenate(y), np.concatenate(y_hat), np.concatenate(overlaps)

        result["balanced_acc"].append(compute_balanced_accuracy(y, y_hat))
        result["roc_auc"].append(compute_roc_auc(y, y_hat))
        result["f1"].append(compute_f1(y, y_hat))
        result["overlap"].append(np.mean(overlaps))

    time.sleep(5)
    for key, value in result.items():
        print(f"{key.capitalize()} mean: {statistics.mean(value)}, std: {statistics.stdev(value)}")


if __name__ == '__main__':
    main()
