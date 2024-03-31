import cv2
import os

from house_detector import HouseDetector
from constants import RAW_IMAGES_PATH

image = cv2.imread(os.path.join(RAW_IMAGES_PATH, 'austin1.tif'))

detector = HouseDetector(model_filename='model1_fcos5.pth')
boxes, labels, scores = detector.detect(image)
