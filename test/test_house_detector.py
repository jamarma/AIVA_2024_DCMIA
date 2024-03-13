import unittest
import numpy as np
import cv2 as cv

from src.app.user.house_detector import HouseDetector


class TestHouseDetector(unittest.TestCase):

    def setUp(self):
        self.detector = HouseDetector()

    def test_predict_returns_list_of_tuples(self):
        image = "test/data/input_example.png"
        result = self.detector.predict(image)

        self.assertIsInstance(result, list)

        for bbox, score in result:
            self.assertIsInstance(bbox, np.ndarray)
            for coord in bbox:
                self.assertIsInstance(coord, float)
            self.assertIsInstance(score, float)

    def test_predict_returns_correct_format(self):
        image = "test/data/input_example.png"
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
        image = "test/data/input_example.png"
        result = self.detector.predict(image)

        for bbox, _ in result:
            img_height, img_width, _ = cv.imread(image).shape

            for coord in bbox:
                self.assertGreaterEqual(coord, 0)
                if coord % 2 == 0:
                    self.assertLessEqual(coord, img_width)
                else:
                    self.assertLessEqual(coord, img_height)


if __name__ == '__main__':
    unittest.main()
