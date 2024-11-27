# Need to add attribution https://challenge.isic-archive.com/data/#2019
# I do not know exactly which since I used https://github.com/laura-rieger/deep-explanation-penalization to download
# Segmentation maps
# https://drive.google.com/drive/folders/1Er2PQMwmDSmg3BThyeu-JKX442OkQJit

from utils import get_project_root
import torchvision
import torch
import skimage as ski
import numpy as np
from dataclasses import dataclass
import tyro
import joblib
import lightning as L
from data_utils import plot_img


@dataclass
class Args:
    num_img: int = 50
    seed: int = 8434
    # https://www.random.org/, Min: 0, Max: 10000, 2024-10-16 07:49:07 UTC
    train: float = 0.6


def load_masks(folder):
    mask_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),
    ])

    masks = dict()

    for file in folder.iterdir():
        name = file.stem.split('_')[1]
        masks[name] = ski.io.imread(file, as_gray=True)
        masks[name] = np.where(masks[name] > 0.5, 1, 0)
        masks[name] = mask_transform(masks[name])

    return masks


def load_img(folder, masks):
    if "benign" in str(folder):
        label = 0
    elif "malignant" in str(folder):
        label = 1
    else:
        raise ValueError("The given folder is invalid")

    img_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    data = []

    for file in folder.iterdir():
        name = file.stem.split('_')[1]
        img = ski.io.imread(file)
        img = img_transform(img)
        if name in masks:
            mask = masks[name]
        else:
            mask = torch.zeros((1, img.shape[1], img.shape[2]), dtype=torch.int64)

        data.append((img, label, mask))

    return data


def split_data(data, idx, used_idx):
    subset = []
    for i in idx:
        assert i not in used_idx, f"id={i} already in a dataset."
        used_idx.add(i)
        subset.append(data[i])
    return subset


def rearrange_data(data):
    return (torch.stack([i[0] for i in data], 0).to(dtype=torch.float32),
            torch.tensor([i[1] for i in data], dtype=torch.float32),
            torch.stack([i[2] for i in data], 0).to(dtype=torch.float32))


def main():
    args = tyro.cli(Args)
    L.seed_everything(args.seed, workers=True, verbose=True)
    data_folder = get_project_root() / 'data/isic'

    # Load data
    masks = load_masks(data_folder / "masks")
    benign = load_img(data_folder / "img/benign", masks)
    malignant = load_img(data_folder / "img/malignant", masks)
    data = benign + malignant

    # Split data
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    used_idx = set()
    trainval = split_data(data, idx[:int(args.train * len(idx))], used_idx)
    x_trainval, y_trainval, mask_trainval = rearrange_data(trainval)
    plot_img(data_folder / "train_examples", x_trainval, y_trainval, mask_trainval, args.num_img)

    with data_folder.joinpath("trainval.joblib").open("wb") as f:
        joblib.dump((x_trainval, y_trainval, mask_trainval), f)

    # Test
    test = split_data(data, idx[int((args.train) * len(idx)):], used_idx)
    x_test, y_test, mask_test = rearrange_data(test)
    plot_img(data_folder / "test_examples", x_test, y_test, mask_test, args.num_img)

    with data_folder.joinpath("test.joblib").open("wb") as f:
        joblib.dump((x_test, y_test, mask_test), f)


if __name__ == '__main__':
    main()
