import shutil
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def unnormalize_input(images: torch.Tensor) -> torch.Tensor:
    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/4
    img_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=list(map(lambda x: 1. / x, STD))),
        transforms.Normalize(mean=list(map(lambda x: -x, MEAN)),
                             std=[1., 1., 1.]),
    ])

    return img_transform(images)


def transform_input(images: list[Image]) -> torch.Tensor:
    # Image size = (b,w,h,c)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    return torch.stack([img_transform(img) for img in images], 0)


def transform_mask(masks: list):
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224),
                          interpolation=transforms.InterpolationMode.NEAREST),
    ])
    return torch.stack([mask_transform(mask) for mask in masks], 0)


def plot_img(folder: Path, x, y, mask, num_img: int) -> None:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

    idx = np.arange(len(x))
    np.random.shuffle(idx)

    # Plot individual images
    img = np.transpose(unnormalize_input(x).numpy(force=True), (0, 2, 3, 1))
    img = np.clip(img, 0, 1)
    mask = np.transpose(mask.numpy(force=True), (0, 2, 3, 1))
    for i in range(num_img):
        fig, ax = plt.subplots(ncols=2, layout="constrained")
        ax[0].imshow(img[idx[i]])
        ax[1].imshow(mask[idx[i]], cmap='gray')
        ax[0].axis('off')
        ax[1].axis('off')
        fig.savefig(folder / f"idx={idx[i]}__y={y[idx[i]]}.png", bbox_inches='tight',
                    pad_inches=0.0)
        plt.close(fig)
