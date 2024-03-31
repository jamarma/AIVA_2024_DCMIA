import numpy as np
import torch

import utils
from object_detectors import FasterRCNN, FCOS, RetinaNet
from constants import CLASSES
from predictions_patch import PredictionsPatch
from predictions_patch_matrix import PredictionsPatchMatrix


class HouseDetector:
    """Class to detect houses in high resolution images"""
    def __init__(self, model_filename: str):
        """
        Initializes the HouseDetector.

        Parameters:
            - model_filename (str): the filename of the model with .pth
            extension. The file must be in models directory.
        """
        self.object_detector = FCOS(len(CLASSES))
        self.object_detector.load_model(model_filename)

    def detect(self, image: np.array, window_size=500, step_size=400) -> (np.array, np.array, np.array):
        """
        Returns the bounding boxes, labels and scores of the houses detected
        in the input image. Uses a sliding window to make the detections.

        Parameters:
            - image (np.array): the input image.
            - window_size (int): the size of sliding window.
            - step_size (int): the size of the step between windows.

        Returns:
            - boxes_arr (np.array): the bounding boxes of detected houses.
            - labels_arr (np.array): the labels of bounding boxes.
            - scores_arr (np.array): the scores of bounding boxes.
        """
        transform = utils.get_transform()
        # Num of sliding window patches in rows and columns
        num_windows_row = (image.shape[0] - window_size) // step_size + 1
        num_windows_cols = (image.shape[1] - window_size) // step_size + 1
        # Initializes the matrix of predictions patches
        patches_matrix = PredictionsPatchMatrix(shape=(num_windows_row, num_windows_cols))
        for (i, j, x, y, window) in self.__sliding_window(image, window_size, step_size):
            # Transforms numpy window to tensor and makes predictions with
            # object detector
            window_tensor = torch.from_numpy(window.transpose((2, 0, 1)))
            window_tensor = transform(window_tensor)
            boxes, labels, scores = self.object_detector.predict(window_tensor)

            # Initializes a patch with predictions
            patch = PredictionsPatch(boxes, labels, scores)
            # Filters de patch predictions by area
            window_area = window_size * window_size
            patch.filter(max_area_boxes=window_area*0.3)

            # Adds the predictions patch to matrix of predictions patches
            patches_matrix.add_patch(patch, indices=(i, j), coordinates=(x, y))
            # Merges the patch with its neighbors to resolve conflicts
            # of predictions at patch edges.
            patches_matrix.merge_patch_with_neighbours(indices=(i, j))

        boxes, labels, scores = patches_matrix.get_predictions()
        boxes_arr = boxes.cpu().detach().numpy()
        labels_arr = labels.cpu().detach().numpy()
        scores_arr = scores.cpu().detach().numpy()
        return boxes_arr, labels_arr, scores_arr

    @staticmethod
    def __sliding_window(image: np.array, window_size: int, step_size: int) -> (int, int, int, int, np.array):
        """
        Generates sliding window patches from an image with specified
        window size and step size.

        Parameters:
            - image (np.array): the input image.
            - window_size (int): the size of the window.
            - step_size (int): the size of the step between windows.

        Yields:
            - i (int): the index of the window in columns.
            - j (int): the index of the window in rows.
            - x (int): the x coordinate of the upper left corner of the
            sliding window inside the image.
            - y (int): the y coordinate of the upper left corner of the
            sliding window inside the image.
            - np.array: the sliding window patch.
        """
        for j, y in enumerate(range(0, image.shape[0] - window_size, step_size)):
            for i, x in enumerate(range(0, image.shape[1] - window_size, step_size)):
                yield i, j, x, y, image[y:y + window_size, x:x + window_size]
