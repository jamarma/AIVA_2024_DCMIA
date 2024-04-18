import cv2
import numpy as np


class HouseDetectorEvaluator:
    def __init__(self):
        """Initializes the HouseDetectorEvaluator."""
        self.boxes_gt = None

    def add_gt_mask(self, mask: np.array):
        """
        Stores the bounding boxes of the ground truth from a
        binary input mask.

        Parameters:
            - mask (np.array): the binary input mask.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, x+w, y+h))
        self.boxes_gt = np.array(boxes)

    def evaluate_num_houses(self, boxes: np.array) -> float:
        """
        Calculates the ratio of houses detected to the number of
        houses in the ground truth.

        Parameters:
            - boxes: the predicted bounding boxes.

        Returns:
            - float: the ratio calculated.
        """
        # Number of detected bounding boxes
        num_detected_boxes = boxes.shape[0]
        # Number of ground truth bounding boxes
        num_mask_boxes = self.boxes_gt.shape[0]

        # Calculate ratio
        if num_detected_boxes > 0:
            return num_detected_boxes / num_mask_boxes
        return 0

    def evaluate_bounding_boxes(self, boxes: np.array, iou_threshold: float):
        """
        Evaluates predicted bounding boxes with respect to ground
        truth using an IoU threshold. Calculates de average
        precision (AP) and average recall (AR).

        Parameters:
            - boxes (np.array): the predicted bounding boxes.
            - iou_threshold (float): IoU threshold to consider a
            detection as true positive.

        Returns:
            - AP (float): the calculated average precision.
            - AR (float): the calculated average recall.
        """
        TP = 0
        FN = 0
        FP = 0

        # Mark ground truth that have already been matched with a prediction
        matched_gt = [False] * self.boxes_gt.shape[0]

        # Iterate over all predictions
        for box in boxes:
            max_iou = 0
            max_iou_index = -1

            # Find the ground truth with the highest IoU
            for i, gt_box in enumerate(self.boxes_gt):
                iou = self.__calculate_iou(box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_index = i

            # If IoU is greater than the threshold, consider as true positive
            if max_iou >= iou_threshold:
                TP += 1
                matched_gt[max_iou_index] = True
            else:
                FP += 1

        # Unmatched elements are considered false negatives
        FN = sum(1 for matched in matched_gt if not matched)

        # Calculates average precision and average recall
        AP = TP / (TP + FP) if (TP + FP) != 0 else 0
        AR = TP / (TP + FN) if (TP + FN) != 0 else 0

        return AP, AR

    @staticmethod
    def __calculate_iou(box1: np.array, box2: np.array):
        """
        Calculate the Intersection over Union (IoU) between two
        bounding boxes.

        Parameters:
            - box1 (np.array): coordinates of the bounding box in
            the format [xmin, ymin, xmax, ymax].
            - box2 (np.array): coordinates of the bounding box in
            the format [xmin, ymin, xmax, ymax].

        Returns:
            - iou (float): IoU value.
        """
        # Coordinates of the intersection (top-left corner and bottom-right corner)
        x1_intersect = max(box1[0], box2[0])
        y1_intersect = max(box1[1], box2[1])
        x2_intersect = min(box1[2], box2[2])
        y2_intersect = min(box1[3], box2[3])

        # Intersection area
        intersection_area = max(0, x2_intersect - x1_intersect + 1) * max(0, y2_intersect - y1_intersect + 1)

        # Areas of the bounding boxes
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # Union area
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area
        return iou



