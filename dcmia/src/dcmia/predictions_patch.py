import torch
from torchvision.ops import nms


class PredictionsPatch:
    """Class that represents the predictions of an image patch."""
    def __init__(self, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor):
        """
        Initializes the PredictionsPatch.

        Parameters:
            - boxes (torch.Tensor): bounding boxes of predictions.
            - labels (torch.Tensor): labels of bounding boxes.
            - scores (torch.Tensor): scores of bounding boxes.
        """
        self.boxes = boxes
        self.labels = labels
        self.scores = scores
        self.num_predictions = boxes.shape[0]

    def merge(self, patch: 'PredictionsPatch') -> 'PredictionsPatch':
        """
        Merges its own predictions with the input patch predictions.
        Resolves conflicts at the edge of the two patches using NMS.
        Once merged, separates the predictions back to their original
        two patches. Returns the input patch after merging.

        Parameters:
            - patch (PredictionsPatch): the input patch to merge.

        Returns:
            - PredictionsPatch: the input patch after merging.
        """
        new_boxes = torch.vstack((self.boxes, patch.boxes))
        new_labels = torch.hstack((self.labels, patch.labels))
        new_scores = torch.hstack((self.scores, patch.scores))

        keep = nms(new_boxes, new_scores, iou_threshold=0.2)
        keep1 = keep[torch.lt(keep, self.num_predictions)]
        keep2 = keep[torch.ge(keep, self.num_predictions)]

        self.boxes, self.labels, self.scores = new_boxes[keep1], new_labels[keep1], new_scores[keep1]
        return PredictionsPatch(new_boxes[keep2], new_labels[keep2], new_scores[keep2])

    def filter(self, max_area_boxes: float):
        """
        Filter the predictions of patch by eliminating those whose
        bounding box area is smaller than the provided input area.

        Parameters:
            - max_area_boxes (float): maximum area allowed for
            bounding boxes.
        """
        box_width = self.boxes[:, 2] - self.boxes[:, 0]
        box_height = self.boxes[:, 3] - self.boxes[:, 1]
        box_areas = box_width * box_height
        self.boxes = self.boxes[box_areas <= max_area_boxes]
        self.labels = self.labels[box_areas <= max_area_boxes]
        self.scores = self.scores[box_areas <= max_area_boxes]
