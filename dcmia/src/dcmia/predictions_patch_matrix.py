import torch
import numpy as np
from torchvision.utils import draw_bounding_boxes

from . import utils
from .constants import CLASSES
from .predictions_patch import PredictionsPatch


class PredictionsPatchMatrix:
    """Class that represents a matrix of PredictionsPatch objects."""
    def __init__(self, shape: (int, int)):
        """
        Initializes the PredictionsPatchMatrix.

        Parameters:
            - shape (int, int): shape of the matrix.
        """
        self.rows = shape[0]
        self.cols = shape[1]
        self.patches = [[None for _ in range(self.cols)] for _ in range(self.rows)]

    def add_patch(self, patch: PredictionsPatch, indices: (int, int), coordinates: (int, int)):
        """
        Adds a PredictionsPatch object in the provided indices.

        Parameters:
            - patch (PredictionsPatch): the PredictionsPatch object to add.
            - indices (int, int): the position (i, j) in which to add the patch
            to the matrix.
            - coordinates (int, int): the coordinates (x, y) of the upper left
            corner of the patch in the original image.
        """
        i, j = indices
        x, y = coordinates
        patch.boxes = patch.boxes + torch.tensor([x, y, x, y]).to(patch.boxes.device)
        self.patches[j][i] = patch

    def merge_patch_with_neighbours(self, indices: (int, int)):
        """
        Merges the patch of the given position with its neighboring patches to
        resolve conflicts at the edges of the patches.

        Parameters:
            - indices (int, int): the position (i, j) of the patch to merge.
        """
        i, j = indices
        patch = self.patches[j][i]
        if patch is None:
            raise ValueError(f'Not patch in desired position (i: {i}, j: {j})')
        if i > 0:
            left_patch = self.patches[j][i-1]
            left_patch = patch.merge(left_patch)
            self.patches[j][i-1] = left_patch
        if j > 0:
            top_patch = self.patches[j-1][i]
            top_patch = patch.merge(top_patch)
            self.patches[j-1][i] = top_patch
        self.patches[j][i] = patch

    def draw(self, image: np.array):
        """
        Draws the predictions of all patches in the input image.

        Parameters:
            - image (np.array): the input image.

        Return:
            - output (np.array): the input image with predictions drawn.
        """
        transform = utils.get_transform()
        img_tensor = torch.from_numpy(image.transpose((2, 0, 1)))
        img_tensor = transform(img_tensor)
        img_tensor = (255.0 * (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())).to(torch.uint8)

        for j in range(self.rows):
            for i in range(self.cols):
                patch = self.patches[j][i]
                if patch is not None:
                    text_labels = [f'{CLASSES[label]}: {score:.3f}' for label, score in zip(patch.labels, patch.scores)]
                    img_tensor = draw_bounding_boxes(img_tensor, patch.boxes, text_labels, colors='blue')
        output = img_tensor.cpu().permute(1, 2, 0).numpy()
        return output

    def get_predictions(self):
        """
        Returns the predictions of all patches.

        Returns:
            - output_boxes (np.array): the bounding boxes of all patches.
            - output_labels (np.array): the labels of all bounding boxes.
            - output_scores (np.array): the scores of all bounding boxes.
        """
        patch = self.patches[0][0]
        if patch is None:
            raise ValueError(f'None in (i: {0}, j:{0}) of patchs matrix')
        output_boxes = patch.boxes.clone()
        output_labels = patch.labels.clone()
        output_scores = patch.scores.clone()
        for j in range(self.rows):
            for i in range(self.cols):
                patch = self.patches[j][i]
                if patch is None:
                    raise ValueError(f'None in (i: {i}, j:{j}) of patchs matrix')
                output_boxes = torch.vstack((output_boxes, patch.boxes))
                output_labels = torch.hstack((output_labels, patch.labels))
                output_scores = torch.hstack((output_scores, patch.scores))
        return output_boxes, output_labels, output_scores



