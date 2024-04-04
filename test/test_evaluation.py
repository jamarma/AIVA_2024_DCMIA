import unittest
import cv2
import numpy as np
from src.house_detector_evaluator import HouseDetectorEvaluator

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = HouseDetectorEvaluator()

        # Example ground truth mask
        self.mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(self.mask, (10, 10), (50, 50), (255), thickness=-1)
        cv2.rectangle(self.mask, (60, 60), (100, 100), (255), thickness=-1)
        cv2.rectangle(self.mask, (80, 80), (120, 120), (255), thickness=-1)
        cv2.rectangle(self.mask, (150, 150), (190, 190), (255), thickness=-1)
        cv2.rectangle(self.mask, (220, 220), (260, 260), (255), thickness=-1)

        # Example predicted bounding boxes
        self.boxes = np.array([
            [10, 10, 50, 50],
            [60, 60, 100, 100],
            [30, 30, 70, 70],
            [80, 80, 120, 120],
            [110, 110, 150, 150]
        ])

    def test_add_gt_mask(self):
        self.evaluator.add_gt_mask(self.mask)
        self.assertEqual(len(self.evaluator.boxes_gt), 4)

    def test_evaluate_num_houses(self):
        self.evaluator.add_gt_mask(self.mask)
        ratio = self.evaluator.evaluate_num_houses(self.boxes)
        self.assertAlmostEqual(ratio, 1.25)

    def test_evaluate_bounding_boxes(self):
        self.evaluator.add_gt_mask(self.mask)
        AP, AR = self.evaluator.evaluate_bounding_boxes(self.boxes, 0.5)
        self.assertAlmostEqual(AP, 0.2)
        self.assertAlmostEqual(AR, 0.25)


if __name__ == '__main__':
    unittest.main()
