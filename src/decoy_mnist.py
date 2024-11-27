# Dataset by https://arxiv.org/abs/1703.03717
import keras
from utils import get_project_root
from lightning import seed_everything
import numpy as np
import random
import tyro
from dataclasses import dataclass
from tqdm import trange
import matplotlib.pyplot as plt
import shutil
import joblib
import torch


def add_decoy(x, y, train: bool, modification: str, seed: int):
    seed_everything(seed, workers=True, verbose=False)
    x_clone = np.copy(x)
    decoy = np.zeros_like(x_clone)
    num_classes = np.max(y)
    decoy_size = 4
    shift = 2

    for i in trange(x_clone.shape[0]):
        if train:
            if modification == "color":
                corner = random.randint(0, 9)
                value = y[i]
            elif modification == "position":
                corner = y[i]
                value = random.randint(0, num_classes)
            else:
                raise ValueError
        else:
            corner = random.randint(0, 9)
            value = random.randint(0, num_classes)
        color = 255 - 25 * value
        match corner:
            case 0:
                x_clone[i, :decoy_size, :decoy_size] = color
                decoy[i, :decoy_size, :decoy_size] = 255
            case 1:
                x_clone[i, :decoy_size, -decoy_size:] = color
                decoy[i, :decoy_size, -decoy_size:] = 255
            case 2:
                x_clone[i, -decoy_size:, :decoy_size] = color
                decoy[i, -decoy_size:, :decoy_size] = 255
            case 3:
                x_clone[i, -decoy_size:, -decoy_size:] = color
                decoy[i, -decoy_size:, -decoy_size:] = 255
            case 4:
                x_clone[i, (shift - 1) * decoy_size: shift * decoy_size, :decoy_size] = color
                decoy[i, (shift - 1) * decoy_size: shift * decoy_size, :decoy_size] = 255
            case 5:
                x_clone[i, :decoy_size, (shift - 1) * decoy_size: shift * decoy_size] = color
                decoy[i, :decoy_size, (shift - 1) * decoy_size: shift * decoy_size] = 255
            case 6:
                x_clone[i, :decoy_size, shift * -decoy_size:(shift - 1) * - decoy_size] = color
                decoy[i, :decoy_size, shift * -decoy_size:(shift - 1) * - decoy_size] = 255
            case 7:
                x_clone[i, (shift - 1) * decoy_size: shift * decoy_size, -decoy_size:] = color
                decoy[i, (shift - 1) * decoy_size: shift * decoy_size, -decoy_size:] = 255
            case 8:
                x_clone[i, -decoy_size:, (shift - 1) * decoy_size:shift * decoy_size] = color
                decoy[i, -decoy_size:, (shift - 1) * decoy_size: shift * decoy_size] = 255
            case 9:
                x_clone[i, shift * -decoy_size:(shift - 1) * -decoy_size, :decoy_size] = color
                decoy[i, shift * -decoy_size:(shift - 1) * -decoy_size, :decoy_size] = 255
            case _:
                raise ValueError

    return x_clone, decoy


def plot_img(x, y, mask, folder, num_plot: int, seed: int):
    seed_everything(seed, workers=True, verbose=False)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()

    samples = random.sample(range(x.shape[0]), num_plot)

    for i in samples:
        fig, ax = plt.subplots(ncols=1, nrows=1, layout="constrained")
        ax.imshow(x[i], cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        fig.savefig(folder / f"id={i}__label={y[i]}.png", bbox_inches='tight',
                    pad_inches=0.0)
        plt.close(fig)

    if mask is not None:
        for i in samples:
            fig, ax = plt.subplots(ncols=1, nrows=1, layout="constrained")
            ax.imshow(mask[i], cmap='gray', vmin=0, vmax=255)
            ax.axis('off')
            fig.savefig(folder / f"id={i}__label={y[i]}__mask.png", bbox_inches='tight',
                        pad_inches=0.0)
            plt.close(fig)

    # Plot large plot
    img = x

    idx = np.arange(len(img))
    np.random.shuffle(idx)

    ncols, nrows = 1, 4

    fixed_idx = []
    used_labels = set()
    for i in idx:
        if y[i] not in used_labels:
            fixed_idx.append(i)
            used_labels.add(y[i])
        if len(fixed_idx) >= ncols * nrows:
            break

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, layout="constrained",
                            figsize=(ncols * 4, nrows * 4))

    for i, ax in enumerate(axs.reshape(-1)):
        ax.imshow(img[fixed_idx[i]], cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
    fig.savefig(folder / f"decoy_mnist.pdf", bbox_inches='tight',
                pad_inches=0.0)


def split_data(x, y, mask, idx, used_idx):
    subset = []
    for i in idx:
        assert i not in used_idx, f"id={i} already in a dataset."
        used_idx.add(i)
        subset.append((x[i], y[i], mask[i]))
    return subset


def rearrange_data(data):
    data = (np.expand_dims(np.stack([i[0] for i in data], 0), 1) / 255.0,
            np.array([i[1] for i in data]),
            np.expand_dims(np.stack([i[2] for i in data], 0), 1) / 255.0)
    data = (torch.tensor(data[0], dtype=torch.float32),
            torch.tensor(data[1], dtype=torch.float32),
            torch.tensor(data[2], dtype=torch.float32))
    return data


@dataclass
class Args:
    modification: str = "color"
    num_plot: int = 50
    seed: int = 0
    train: int = 0.8


def main():
    args = tyro.cli(Args)
    assert args.modification in ("color", "position"), "Modification must be either color or position."
    seed_everything(args.seed, workers=True, verbose=True)
    # Load the dataset
    mnist_folder = get_project_root() / "data/mnist"
    if mnist_folder.exists():
        shutil.rmtree(mnist_folder)
    mnist_folder.mkdir(parents=True)
    (x_trainval, y_trainval), (x_test, y_test) = keras.datasets.mnist.load_data(path=mnist_folder / "mnist.npz")

    plot_img(x_trainval, y_trainval, None, mnist_folder / "train", args.num_plot, args.seed)
    plot_img(x_test, y_test, None, mnist_folder / "test", args.num_plot, args.seed)

    # Make decoy version
    x_trainval_decoy, mask_trainval = add_decoy(x_trainval, y_trainval, True, args.modification, args.seed)
    x_test_decoy, mask_test = add_decoy(x_test, y_test, False, args.modification, args.seed)

    decoy_mnist_folder = get_project_root() / f"data/decoy_mnist_{args.modification}"
    if decoy_mnist_folder.exists():
        shutil.rmtree(decoy_mnist_folder)
    decoy_mnist_folder.mkdir(parents=True)

    plot_img(x_trainval_decoy, y_trainval, mask_trainval, decoy_mnist_folder / "train", args.num_plot, args.seed)
    plot_img(x_test_decoy, y_test, mask_test, decoy_mnist_folder / "test", args.num_plot, args.seed)

    x_trainval_decoy = np.expand_dims(x_trainval_decoy, 1) / 255.0
    mask_trainval = np.expand_dims(mask_trainval, 1) / 255.0

    x_trainval_decoy = torch.tensor(x_trainval_decoy, dtype=torch.float32)
    y_trainval = torch.tensor(y_trainval, dtype=torch.float32)
    mask_trainval = torch.tensor(mask_trainval, dtype=torch.float32)

    # Save decoy data
    with decoy_mnist_folder.joinpath("trainval.joblib").open("wb") as f:
        joblib.dump((x_trainval_decoy, y_trainval, mask_trainval), f)

    x_test_decoy = np.expand_dims(x_test_decoy, 1) / 255.0
    mask_test = np.expand_dims(mask_test, 1) / 255.0

    x_test_decoy = torch.tensor(x_test_decoy, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    mask_test = torch.tensor(mask_test, dtype=torch.float32)

    with decoy_mnist_folder.joinpath("test.joblib").open("wb") as f:
        joblib.dump((x_test_decoy, y_test, mask_test), f)


if __name__ == '__main__':
    main()
