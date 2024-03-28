import os

from houses_dataset import HousesDataset
import torch
from house_detector import ModelFactory
from torch.utils.data import DataLoader
import utils
from constants import (
    CLASSES, TRAIN_DIR, VAL_DIR, TEST_DIR
)
from engine import train_one_epoch, evaluate

train_dataset = HousesDataset(TRAIN_DIR, CLASSES, transforms=utils.get_train_transform())
val_dataset = HousesDataset(VAL_DIR, CLASSES, transforms=utils.get_test_transform())
test_dataset = HousesDataset(TEST_DIR, CLASSES, transforms=utils.get_test_transform())
train_loader = DataLoader(
    train_dataset,
    batch_size=10,
    shuffle=True,
    num_workers=0,
    collate_fn=utils.collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=10,
    shuffle=True,
    num_workers=0,
    collate_fn=utils.collate_fn
)
test_loader = DataLoader(
    test_dataset,
    batch_size=10,
    shuffle=True,
    num_workers=0,
    collate_fn=utils.collate_fn
)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 1

# get the model using our helper function
model = ModelFactory.create_model(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.001,
    momentum=0.5,
    weight_decay=0.001
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it just for 2 epochs
num_epochs = 3

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, test_loader, device=device)


output_dir = "model"
model_filename = "trained_model.pth"
model_path = os.path.join(output_dir, model_filename)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

torch.save(model, model_path)
print("Training completed successfully")
