import torch
import os
import torchvision
from functools import partial
from torchvision.ops import nms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.fcos import FCOSClassificationHead
from abc import ABC, abstractmethod

from .torchvision_sources.engine import train_one_epoch, evaluate
from . import constants


class ObjectDetector(ABC):
    def __init__(self, num_classes: int, train_data_loader=None, val_data_loader=None) -> None:
        """
        Initializes the ObjectDetector.

        Parameters:
            - num_classes (int): Number of classes for detection.
            - train_data_loader: DataLoader for training data.
            - val_data_loader: DataLoader for validation data.
        """
        self.model = self._model_instance(num_classes)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

    def train(self, num_epochs: int, lr: float) -> None:
        """
        Trains the object detection model.

        Parameters:
            - num_epochs (int): Number of epochs for training.
            - lr (float): Learning rate for training.
        """
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=0.9,
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

    def predict(self, image: torch.Tensor, threshold: float = 0.5) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Predicts bounding boxes, labels, and scores for given image.

        Parameters:
            - image (torch.Tensor): Input image tensor.
            - threshold (float): Detection threshold.

        Returns:
            - labels (torch.Tensor): Predicted labels.
            - boxes (torch.Tensor): Predicted bounding boxes.
            - scores (torch.Tensor): Predicted scores.
        """
        image = image.to(self.device)
        with torch.no_grad():
            pred = self.model([image, ])[0]

        # Filter pred by threshold
        mask = pred['scores'] >= threshold
        pred = {key: value[mask] for key, value in pred.items()}

        # Apply non-maximum suppression to filter overlapping detections
        keep = nms(pred['boxes'], pred['scores'], iou_threshold=0.2)
        pred = {key: value[keep] for key, value in pred.items()}

        return pred['boxes'], pred['labels'], pred['scores']

    def save_model(self, filename: str) -> None:
        """
        Saves the trained model to a file.

        Parameters:
            - filename (str): Name of the file to save.
        """
        path = os.path.join(constants.MODELS_PATH, filename)
        torch.save(self.model.state_dict(), path)

    def load_model(self, filename: str) -> None:
        """
        Loads a trained model from a file.

        Parameters:
            - filename (str): Name of the file to load.
        """
        path = os.path.join(constants.MODELS_PATH, filename)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    @staticmethod
    @abstractmethod
    def _model_instance(num_classes: int):
        """
        Abstract method to instantiate the specific model.

        Parameters:
            - num_classes (int): Number of classes for detection.

        Returns:
            - model: Instance of the specific model.
        """
        return NotImplemented


class FasterRCNN(ObjectDetector):
    def __init__(self, num_classes: int, train_data_loader=None, val_data_loader=None) -> None:
        """
        Initializes the FasterRCNN object detector.

        Parameters:
            - num_classes (int): Number of classes for detection.
            - train_data_loader: DataLoader for training data.
            - val_data_loader: DataLoader for validation data.
        """
        super().__init__(
            num_classes,
            train_data_loader,
            val_data_loader
        )

    @staticmethod
    def _model_instance(num_classes: int):
        """
        Instantiates the Faster R-CNN model.

        Parameters:
            - num_classes (int): Number of classes for detection.

        Returns:
            - model: Faster R-CNN model instance.
        """
        # load a model pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model


class RetinaNet(ObjectDetector):
    def __init__(self, num_classes: int, train_data_loader=None, val_data_loader=None) -> None:
        """
        Initializes the RetinaNet object detector.

        Parameters:
            - num_classes (int): Number of classes for detection.
            - train_data_loader: DataLoader for training data.
            - val_data_loader: DataLoader for validation data.
        """
        super().__init__(
            num_classes,
            train_data_loader,
            val_data_loader,
        )

    @staticmethod
    def _model_instance(num_classes: int):
        """
        Instantiates the RetinaNet model.

        Parameters:
            - num_classes (int): Number of classes for detection.

        Returns:
            - model: RetinaNet model instance.
        """
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)

        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
        return model


class FCOS(ObjectDetector):
    def __init__(self, num_classes: int, train_data_loader=None, val_data_loader=None) -> None:
        """
        Initializes the FCOS object detector.

        Parameters:
            - train_data_loader: DataLoader for training data.
            - val_data_loader: DataLoader for validation data.
            - num_classes (int): Number of classes for detection.
        """
        super().__init__(
            num_classes,
            train_data_loader,
            val_data_loader,
        )

    @staticmethod
    def _model_instance(num_classes: int, min_size: int = 640, max_size: int = 640):
        """
        Instantiates the FCOS model.

        Parameters:
            - num_classes (int): Number of classes for detection.
            - min_size (int): Minimum input size.
            - max_size (int): Maximum input size.

        Returns:
            - model: FCOS model instance.
        """
        model = torchvision.models.detection.fcos_resnet50_fpn(weights='DEFAULT')
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = FCOSClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
        model.transform.min_size = (min_size,)
        model.transform.max_size = max_size
        for param in model.parameters():
            param.requires_grad = True
        return model
