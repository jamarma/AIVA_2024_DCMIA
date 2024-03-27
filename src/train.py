from torch.utils.data import DataLoader

import utils
from constants import (
    CLASSES, TRAIN_DIR, TEST_DIR
)
from houses_dataset import HousesDataset


train_dataset = HousesDataset(TRAIN_DIR, CLASSES, transforms=utils.get_train_transform())
test_dataset = HousesDataset(TEST_DIR, CLASSES, transforms=utils.get_test_transform())
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

print(f"Number of training samples: {len(train_dataset)}")
image, target = train_dataset[10]
utils.visualize_dataset_sample(image, target)
