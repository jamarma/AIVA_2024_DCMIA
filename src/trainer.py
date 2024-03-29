from typing import List, Tuple
import os
import torch
from torch.utils.data import DataLoader
from houses_dataset import HousesDataset
from house_detector import ModelFactory
from src.detection.engine import train_one_epoch, evaluate
import utils
from constants import CLASSES, TRAIN_DIR, VAL_DIR, TEST_DIR

class Trainer:
    def __init__(self, train_dir: str, val_dir: str, test_dir: str, classes: List[str], num_epochs: int = 3, batch_size: int = 10, lr: float = 0.001, momentum: float = 0.5, weight_decay: float = 0.001) -> None:
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.classes = classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def train_model(self) -> None:
        train_dataset = HousesDataset(self.train_dir, self.classes, transforms=utils.get_train_transform())
        val_dataset = HousesDataset(self.val_dir, self.classes, transforms=utils.get_test_transform())
        test_dataset = HousesDataset(self.test_dir, self.classes, transforms=utils.get_test_transform())

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)

        model = ModelFactory.create_model(num_classes=1)
        model.to(self.device)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(self.num_epochs):
            train_one_epoch(model, optimizer, train_loader, self.device, epoch, print_freq=10)
            lr_scheduler.step()
            evaluate(model, val_loader, device=self.device)

        output_dir = "model"
        model_filename = "trained_model.pth"
        model_path = os.path.join(output_dir, model_filename)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(model, model_path)
        print("Training completed successfully")


if __name__ == "__main__":
    # Hiperparámetros
    num_epochs = 3
    batch_size = 10
    lr = 0.001
    momentum = 0.5  # Not higher
    weight_decay = 0.0005

    trainer = Trainer(
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        CLASSES,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    trainer.train_model()
