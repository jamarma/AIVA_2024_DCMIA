import unittest
import cv2
from src.evaluation import Evaluator

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        # Example data
        self.mask_boxes = [(10, 10, 50, 50), (60, 60, 100, 100), (80, 80, 120, 120), (190, 190, 230, 230), (240, 240, 280, 280)]

        self.boxes = [(10, 10, 50, 50), (60, 60, 100, 100), (30, 30, 70, 70), (80, 80, 120, 120), (220, 220, 260, 260)]

    def test_calculate_iou(self):
        box1 = self.boxes[0]
        box2 = self.mask_boxes[0]
        iou = Evaluator.calculate_iou(box1, box2)

        self.assertAlmostEqual(iou, 1)

    def test_evaluate_bounding_boxes(self):
        pred_boxes = self.boxes
        gt_boxes = self.mask_boxes
        iou_threshold = 0.5
        TP, FN, FP = Evaluator.evaluate_bounding_boxes(pred_boxes, gt_boxes, iou_threshold)

        self.assertEqual(TP, 3)
        self.assertEqual(FN, 2)
        self.assertEqual(FP, 2)

if __name__ == '__main__':
    unittest.main()
