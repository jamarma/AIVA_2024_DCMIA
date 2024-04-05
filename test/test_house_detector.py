import unittest
import cv2
import numpy as np
import sys
sys.path.insert(0, '../src')

from house_detector import HouseDetector


class TestHouseDetector(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Example image
        self.image = cv2.imread("data/austin1.tif")
        self.void_image = np.zeros((5000, 5000, 3), dtype=np.uint8)

        # Example model filename
        self.model_filename = "../models/model1_fcos5.pth"

        # Create a HouseDetector instance
        self.house_detector = HouseDetector(self.model_filename)

        # Example window size and step size
        window_size = 500
        step_size = 400

        # Call the detect method
        self.boxes_arr, self.labels_arr, self.scores_arr = self.house_detector.detect(self.image, window_size, step_size)

    def test_detect_check_format(self):
        # Ensure the arrays are correct
        self.assertEqual(self.boxes_arr.shape[1], 4)  # Each bounding box should have 4 coordinates
        self.assertEqual(self.labels_arr[1], 1)  # Each label should be a single value
        self.assertIsInstance(self.scores_arr[0], np.float32)  # Each score should be a single value

    def test_detect_check_valid_values(self):
        # Ensure the values are within a valid range
        self.assertTrue(np.all(self.boxes_arr >= 0))
        self.assertTrue(np.all(self.scores_arr >= 0))
        self.assertTrue(np.all(self.scores_arr <= 1))
        self.assertTrue(np.all(self.labels_arr == 1))

    def test_detect_check_consistent(self):
        # Ensure the number of boxes, labels, and scores are consistent
        self.assertEqual(self.boxes_arr.shape[0], self.labels_arr.shape[0])
        self.assertEqual(self.boxes_arr.shape[0], self.scores_arr.shape[0])

    def test_detect_void_image(self):
        # Create a HouseDetector instance
        house_detector = HouseDetector(self.model_filename)

        # Example window size and step size
        window_size = 500
        step_size = 400

        # Call the detect method
        boxes_arr, labels_arr, scores_arr = house_detector.detect(self.void_image, window_size, step_size)

        # Ensure the arrays are correct
        self.assertEqual(boxes_arr.shape[0], 0)

        # Ensure the number of boxes, labels, and scores are consistent
        self.assertEqual(boxes_arr.shape[0], labels_arr.shape[0])
        self.assertEqual(boxes_arr.shape[0], scores_arr.shape[0])


if __name__ == '__main__':
    unittest.main()