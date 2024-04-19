import cv2
import argparse

from src.dcmia import utils
from src.dcmia.house_detector import HouseDetector
from src.dcmia.house_detector_evaluator import HouseDetectorEvaluator

#  python main.py --image_path data/raw/test/images/austin1.tif --mask_path data/raw/test/masks/austin1.tif


def main(image_path: str, mask_path=None, output_path = None):
    image = cv2.imread(image_path)

    detector = HouseDetector(model_filename='model1_fcos5.pth')
    evaluator = HouseDetectorEvaluator()
    boxes, labels, scores = detector.detect(image)

    print('Number of houses detected: ', boxes.shape[0])
    if output_path is not None:
        output = utils.draw_boxes(image, boxes)
        cv2.imwrite(output_path, output)
        print(f'Output image saved to {output_path}!')

    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        evaluator.add_gt_mask(mask)
        ratio = evaluator.evaluate_num_houses(boxes)
        print(f'Ratio of the number of houses detected: {ratio:.2f}')

        print("Evaluating bounding boxes...")
        iou_thresholds = [0.5, 0.75, 0.9]
        for iou in iou_thresholds:
            AP, AR = evaluator.evaluate_bounding_boxes(boxes, iou_threshold=iou)
            print(f'Average Precision (AP) @[ IoU={iou} ] = {AP:.3f}')
            print(f'Average Recall (AP) @[ IoU={iou} ] = {AR:.3f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='Path to the input image', required=True)
    parser.add_argument('--mask_path', type=str, help='Path to the binary mask ground truth')
    parser.add_argument('--output_path', type=lambda x: utils.is_valid_output_image(parser, x),
                        help='Output image path (.png) with boxes drawn of detected houses')
    args = parser.parse_args()

    main(args.image_path, args.mask_path, args.output_path)

