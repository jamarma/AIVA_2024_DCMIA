import numpy as np
import torch
import os
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from abc import ABC, abstractmethod

from torchvision_sources.engine import train_one_epoch, evaluate
import constants


class ObjectDetector(ABC):
    def __init__(self, train_data_loader, val_data_loader, num_classes):
        self.model = self._model_instance(num_classes)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

    def train(self, num_epochs, lr):
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=0.5,
            weight_decay=0.0005
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.model, optimizer, self.train_data_loader, self.device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(self.model, self.val_data_loader, device=self.device)

    def predict(self, image: np.array):
        pass

    def save_model(self, filename: str):
        path = os.path.join(constants.MODELS_PATH, filename)
        torch.save(self.model.state_dict(), path)

    def load_model(self, filename: str):
        path = os.path.join(constants.MODELS_PATH, filename)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    @staticmethod
    @abstractmethod
    def _model_instance(num_classes: int):
        return NotImplemented


class FasterRCNN(ObjectDetector):
    def __init__(self, train_data_loader, val_data_loader, num_classes):
        super().__init__(
            train_data_loader,
            val_data_loader,
            num_classes
        )

    @staticmethod
    def _model_instance(num_classes: int):
        # load a model pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model


