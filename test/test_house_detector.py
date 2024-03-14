import unittest
import numpy as np
import cv2 as cv

from src.app.user.house_detector import HouseDetector
from src.house_detector.utils.io_utils import get_bounding_boxes


class TestHouseDetector(unittest.TestCase):

    def setUp(self):
        self.detector = HouseDetector()

    def test_predict_returns_list_of_tuples(self):
        image = "test/data/austin1.tif"
        result = self.detector.predict(image)

        self.assertIsInstance(result, list)

        for bbox, score in result:
            self.assertIsInstance(bbox, np.ndarray)
            for coord in bbox:
                self.assertIsInstance(coord, float)
            self.assertIsInstance(score, float)

    def test_predict_returns_correct_format(self):
        image = "test/data/austin1.tif"
        result = self.detector.predict(image)

        for bbox, score in result:
            self.assertEqual(len(bbox), 4)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)

    def test_predict_handles_empty_image(self):
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.detector.predict(empty_image)

        self.assertEqual(len(result), 0)

    def test_predict_coordinates_within_image_size(self):
        image = "test/data/austin1.tif"
        result = self.detector.predict(image)

        for bbox, _ in result:
            img_height, img_width, _ = cv.imread(image).shape

            for coord in bbox:
                self.assertGreaterEqual(coord, 0)
                if coord % 2 == 0:
                    self.assertLessEqual(coord, img_width)
                else:
                    self.assertLessEqual(coord, img_height)

    def test_predict_performance(self):
        image = "test/data/austin1.tif"
        gt = "test/data/gt_austin1.tif"

        bounding_boxes_gt = get_bounding_boxes(gt)
        bounding_boxes_pred = self.detector.predict(image)

        accuracy = len(bounding_boxes_pred) / len(bounding_boxes_gt) * 100
        self.assertGreaterEqual(accuracy, 90)

    def test_predict_less_equal_groundtruth(self):
        image = "test/data/austin1.tif"
        gt = "test/data/gt_austin1.tif"

        bounding_boxes_gt = get_bounding_boxes(gt)
        bounding_boxes_pred = self.detector.predict(image)

        self.assertLessEqual(len(bounding_boxes_pred), len(bounding_boxes_gt))


if __name__ == '__main__':
    unittest.main()
