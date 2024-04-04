import unittest
import cv2
import numpy as np
import torch
from house_detector_evaluator import HouseDetectorEvaluator
from predictions_patch_matrix import PredictionsPatchMatrix
from predictions_patch import PredictionsPatch


class TestPredictionsPatchMatrix(unittest.TestCase):
    def setUp(self):
        # Create an example PredictionsPatch
        boxes = torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]])
        labels = torch.tensor([1, 1])
        scores = torch.tensor([0.9, 0.8])
        self.patch = PredictionsPatch(boxes, labels, scores)

        # Create an example PredictionsPatchMatrix
        self.matrix = PredictionsPatchMatrix(shape=(2, 1))

    def test_add_patch(self):
        # Add the patch to the matrix
        self.matrix.add_patch(self.patch, indices=(0, 0), coordinates=(0, 0))

        # Check if the patch has been added correctly
        self.assertEqual(self.matrix.patches[0][0], self.patch)

    def test_get_predictions(self):
        # Add the patch to the matrix
        self.matrix.add_patch(self.patch, indices=(0, 0), coordinates=(0, 0))
        self.matrix.add_patch(self.patch, indices=(0, 1), coordinates=(0, 0))
        # Get the predictions from the matrix
        output_boxes, output_labels, output_scores = self.matrix.get_predictions()

        # Check if the predictions have been retrieved correctly
        self.assertEqual(output_boxes.shape, (6, 4))
        self.assertEqual(output_labels.shape, (6,))
        self.assertEqual(output_scores.shape, (6,))


if __name__ == '__main__':
    unittest.main()
