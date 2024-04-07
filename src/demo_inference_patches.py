import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

import utils
from houses_dataset import HousesDataset
from object_detectors import FasterRCNN, RetinaNet, FCOS
from constants import CLASSES, TEST_DATA_PATH

# Test images
test_dataset = HousesDataset(TEST_DATA_PATH, CLASSES, transforms=utils.get_transform())
# IMPORTANT: Choose the desired test image!
image, _ = test_dataset[4]

# Loads trained model and makes prediction
object_detector = FCOS(len(CLASSES))
object_detector.load_model('model1_fcos5.pth')
boxes, labels, scores = object_detector.predict(image)

# Normalizes image
image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)

# Draws bounding boxes
text_labels = [f'{CLASSES[label]}: {score:.3f}' for label, score in zip(labels, scores)]
output_image = draw_bounding_boxes(image, boxes, text_labels, colors='red')

plt.figure()
plt.imshow(output_image.cpu().permute(1, 2, 0))
plt.show()
