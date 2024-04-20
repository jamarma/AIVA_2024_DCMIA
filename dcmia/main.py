import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.dcmia import utils
from src.dcmia.house_detector import HouseDetector
from src.dcmia.house_detector_evaluator import HouseDetectorEvaluator
from src.dcmia.constants import SCORE_THRESHOLD


def main(image_path: str, mask_path=None, output_path=None):
    # Reads image
    image = cv2.imread(image_path)

    # Initializes the house detector and it makes the detections
    detector = HouseDetector(model_filename='model1_fcos5.pth')
    boxes, labels, scores = detector.detect(image, SCORE_THRESHOLD)
    print('Number of houses detected: ', boxes.shape[0])
    # Save detections if output_path is provided by user
    if output_path is not None:
        output = utils.draw_boxes(image, boxes)
        cv2.imwrite(output_path, output)
        print(f'Output image saved to {output_path}!')

    # If the ground truth is provided, the evaluation process begins.
    if mask_path:
        # Read de binary mask with ground truth
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Initializes the evaluator
        evaluator = HouseDetectorEvaluator()
        # Adds ground truth to evaluator
        evaluator.add_gt_mask(mask)

        # Calculates metrics for a defined score threshold and iou_threshold
        ratio = evaluator.evaluate_num_houses(boxes)
        TP, FN, FP = evaluator.confusion_matrix(boxes, 0.5)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        print(f'\nMETRICS FOR SCORE THRESHOLD = {SCORE_THRESHOLD}:')
        print(f'> Ratio of the number of houses detected: {ratio:.2f}')
        print(f'> Precision: {precision:.2f}')
        print(f'> Recall: {recall:.2f}')

        # Calculates de precision-recall curve and Average Precision (AP)
        # for a list of iou_thresholds
        iou_thresholds = [0.5]
        for iou in iou_thresholds:
            print(f'\nTHE CALCULATION OF PRECISION-RECALL CURVE BEGINS (IoU = {iou})!')
            precision, recall = evaluator.precision_recall_curve(detector, image, iou)
            AP = np.mean(precision)
            print(f'Average Precision (AP) @[ IoU={iou} ] = {AP:.3f}')
            utils.plot_precision_recall_curve(precision, recall)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='Path to the input image', required=True)
    parser.add_argument('--mask_path', type=str, help='Path to the binary mask ground truth')
    parser.add_argument('--output_path', type=lambda x: utils.is_valid_output_image(parser, x),
                        help='Output image path (.png) with boxes drawn of detected houses')
    args = parser.parse_args()

    main(args.image_path, args.mask_path, args.output_path)

