from torch.utils.data import DataLoader
from houses_dataset import HousesDataset
import utils

from constants import (CLASSES, TRAIN_DATA_PATH, VAL_DATA_PATH,
                       BATCH_SIZE, EPOCHS, LEARNING_RATE)
from object_detectors import FasterRCNN, RetinaNet

if __name__ == "__main__":
    # Loads train and validation data
    train_dataset = HousesDataset(TRAIN_DATA_PATH, CLASSES, transforms=utils.get_transform(train=False))
    val_dataset = HousesDataset(VAL_DATA_PATH, CLASSES, transforms=utils.get_transform(train=False))
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)

    # Object detector training
    object_detector = RetinaNet(len(CLASSES), train_data_loader, val_data_loader)
    object_detector.train(num_epochs=EPOCHS, lr=LEARNING_RATE)
    object_detector.save_model('model1.pth')

