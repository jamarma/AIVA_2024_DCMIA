import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision.transforms import v2 as T
from typing import List, Tuple


def get_transform() -> T.Compose:
    """
    Returns a composition of transformations based on whether it's for training or not.

    Parameters:
        - train (bool): Indicates whether the transformation is for training or not.

    Returns:
        - transforms (T.Compose): Composition of transformations.
    """
    transforms = []
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Collates a batch of samples.

    Parameters:
        - batch (List[Tuple]): List of samples.

    Returns:
        - batch (Tuple): Tuple of samples.
    """
    return tuple(zip(*batch))


