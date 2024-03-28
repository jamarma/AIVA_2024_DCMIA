import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import nn
from typing import Any


class ModelFactory:
    @staticmethod
    def create_model(num_classes: int) -> Any:
        # Carga el modelo Faster RCNN pre-entrenado
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # Obtiene el número de características de entrada
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Define una nueva cabeza para el detector con el número requerido de clases
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model