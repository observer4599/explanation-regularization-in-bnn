import joblib
import torch
import torchvision
from dataclasses import dataclass
import tyro
from utils import get_project_root
from data_utils import transform_input, transform_mask, plot_img
import lightning as L


@dataclass
class Args:
    num_img: int = 50
    seed: int = 0


def pet_mask_transform(mask):
    values = torch.unique(mask)

    mask[mask == values[0]] = 0
    mask[mask == values[1]] = 1
    mask[mask == values[2]] = 0

    return mask


def main():
    args = tyro.cli(Args)
    L.seed_everything(args.seed, workers=True, verbose=True)
    data_folder = get_project_root() / "data"

    # Load the training dataset
    trainval = torchvision.datasets.OxfordIIITPet(data_folder, "trainval", download=True,
                                                  target_types=["segmentation", "category"])

    x_trainval = transform_input([trainval[i][0] for i in range(len(trainval))])
    y_trainval = torch.tensor([trainval[i][1][1] for i in range(len(trainval))], dtype=torch.float32)
    mask_trainval = pet_mask_transform(transform_mask([trainval[i][1][0] for i in range(len(trainval))]))

    plot_img((data_folder / "oxford-iiit-pet/trainval_examples"), x_trainval, y_trainval, mask_trainval,
             args.num_img)

    with (data_folder / "oxford-iiit-pet").joinpath("trainval.joblib").open("wb") as f:
        joblib.dump((x_trainval, y_trainval, mask_trainval), f)

    # Load the training dataset
    test = torchvision.datasets.OxfordIIITPet(data_folder, "test", download=True,
                                              target_types=["segmentation", "category"])

    x_test = transform_input([test[i][0] for i in range(len(test))])
    y_test = torch.tensor([test[i][1][1] for i in range(len(test))], dtype=torch.float32)
    mask_test = pet_mask_transform(transform_mask([test[i][1][0] for i in range(len(test))]))

    plot_img((data_folder / "oxford-iiit-pet/test_examples"), x_test, y_test, mask_test,
             args.num_img)

    with (data_folder / "oxford-iiit-pet").joinpath("test.joblib").open("wb") as f:
        joblib.dump((x_test, y_test, mask_test), f)


if __name__ == '__main__':
    main()
