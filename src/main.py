import cv2
import argparse

from house_detector import HouseDetector
from constants import RAW_IMAGES_PATH, RAW_MASKS_PATH
from evaluation import Evaluator

#  python main.py --image_path ../data/raw/images/austin1.tif --mask_path ../data/raw/masks/austin1.tif

def main(image_path, mask_path=None):
    image = cv2.imread(image_path)

    detector = HouseDetector(model_filename='model1_fcos5.pth')
    evaluator = Evaluator()
    boxes, labels, scores = detector.detect(image)

    if mask_path:
        mask = cv2.imread(mask_path)
        precision, mask_boxes = evaluator.evaluate_num_houses(mask, boxes)

        iou_thresholds = [0.5, 0.75, 0.9]
        for iou in iou_thresholds:
            TP, FN, FP = evaluator.evaluate_bounding_boxes(boxes, mask_boxes, iou)
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            print(f"Average Precision (AP) @[ IoU={iou} ] = {precision:.3f}")
    else:
        print("Boxes:", boxes)
        print("Labels:", labels)
        print("Scores:", scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='House Detection Evaluation')
    parser.add_argument('--image_path', type=str, help='Path to the input image', required=True)

    parser.add_argument('--mask_path', type=str, help='Path to the mask image')

    args = parser.parse_args()

    main(args.image_path, args.mask_path)

