import shutil

import joblib
import numpy as np
import torch
import torchvision
from dataclasses import dataclass
from pathlib import Path
import tyro
from utils import get_project_root
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from data_utils import transform_input, transform_mask, plot_img
import lightning as L


@dataclass
class Args:
    num_img: int = 50
    seed: int = 0


def sbd_transform_mask(masks):
    # Bird: 3, Bus: 6, Cat: 8, Dog: 12, Horse: 13
    valid_labels = np.array([3, 6, 8, 12, 13])
    idx = []
    y = []
    transformed_masks = []

    for i in range(len(masks)):
        mask = np.array(masks[i])
        labels = np.unique(mask)
        present = np.intersect1d(labels, valid_labels)
        if len(present) != 1:
            continue
        idx.append(i)
        # Find label
        y.append(present[0])

        transformed_masks.append(np.where(mask == y[-1], 0, 255))

    return torch.tensor(idx), transformed_masks, np.array(y)


def main():
    args = tyro.cli(Args)
    L.seed_everything(args.seed, workers=True, verbose=True)
    data_folder = get_project_root() / "data/sbd"
    download = not data_folder.exists()

    trainval = torchvision.datasets.SBDataset(data_folder, "train", download=download, mode="segmentation")
    idx_trainval, mask_trainval, y_trainval = sbd_transform_mask([trainval[i][1] for i in range(len(trainval))])
    x_trainval = transform_input([trainval[i][0] for i in range(len(trainval))])[idx_trainval]
    mask_trainval = transform_mask(mask_trainval) / 255

    le = LabelEncoder()
    le.fit(y_trainval)
    y_trainval = torch.tensor(le.transform(y_trainval), dtype=torch.float32)

    plot_img(data_folder / "trainval_examples", x_trainval, y_trainval, mask_trainval, args.num_img)

    with data_folder.joinpath("trainval.joblib").open("wb") as f:
        joblib.dump((x_trainval, y_trainval, mask_trainval), f)

    download = not data_folder.exists()
    test = torchvision.datasets.SBDataset(data_folder, "val", download=download, mode="segmentation")
    idx_test, mask_test, y_test = sbd_transform_mask([test[i][1] for i in range(len(test))])
    x_test = transform_input([test[i][0] for i in range(len(test))])[idx_test]
    mask_test = transform_mask(mask_test) / 255

    y_test = torch.tensor(le.transform(y_test), dtype=torch.float32)

    plot_img(data_folder / "test_examples", x_test, y_test, mask_test, args.num_img)

    with data_folder.joinpath("test.joblib").open("wb") as f:
        joblib.dump((x_test, y_test, mask_test), f)


if __name__ == '__main__':
    main()
