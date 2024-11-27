from pathlib import Path
import random
import numpy as np
import torch


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def seed_everything(seed: int, torch_deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
