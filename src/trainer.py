"""from house_detector import ModelFactory
from houses_dataset import HousesDataset
from utils import Averager, save_model
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt
import time

from torch.utils.data import DataLoader
import utils
from constants import (
    CLASSES, TRAIN_DIR, VAL_DIR, TEST_DIR
)

plt.style.use('ggplot')

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 1
NUM_EPOCHS = 10
OUT_DIR = "src/output"


def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list

    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        # targets = [targets]
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        # targets = [{k: v.to(DEVICE) for k, v in targets.items()}]
        loss_dict = model(images, targets)
        print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list


# function for running validation iterations
def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list

    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list


if __name__ == '__main__':
    train_dataset = HousesDataset(TRAIN_DIR, CLASSES, transforms=utils.get_train_transform())
    val_dataset = HousesDataset(VAL_DIR, CLASSES, transforms=utils.get_test_transform())
    test_dataset = HousesDataset(TEST_DIR, CLASSES, transforms=utils.get_test_transform())
    train_loader = DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=0,
        collate_fn = utils.collate_fn
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
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(test_dataset)}\n")
    # initialize the model and move to the computation device
    model = ModelFactory.create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []
    # name to save the trained model with
    MODEL_NAME = 'model'
    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch + 1} of {NUM_EPOCHS}")
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()
        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(val_loader, model)
        print(f"Epoch #{epoch + 1} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch + 1} validation loss: {val_loss_hist.value:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        # save the current epoch model
        save_model(epoch, model, optimizer)
        # sleep for 5 seconds after each epoch
        time.sleep(5)"""


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

"""# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=utils.collate_fn
)"""

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

print("That's it!")
