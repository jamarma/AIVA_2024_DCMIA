import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision.transforms import v2 as T


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.Resize(500, 500),
        A.RandomRotate90(0.5),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def get_test_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })
