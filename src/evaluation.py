import cv2
import numpy as np


class Evaluator:
    @staticmethod
    def evaluate_num_houses(mask, boxes):
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Calculate the number of detected bounding boxes
        num_detected_bboxes = len(boxes)

        # Calculate the number of ground truth bounding boxes from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for contour in contours:
            # Calcular la bounding box para el contorno
            x, y, w, h = cv2.boundingRect(contour)

            bboxes.append((x, y, x + w, y + h))
        num_mask_bboxes = len(contours)

        # Calculate precision
        if num_detected_bboxes > 0:
            precision = (num_detected_bboxes / num_mask_bboxes) * 100
        else:
            precision = 0.0
        """
        # Draw bounding boxes on the mask
        for bbox in boxes:
            x, y, w, h = bbox
            cv2.rectangle(mask, (int(x), int(y)), (int(w), int(h)), (0, 0, 255), 2)
        """
        return precision, bboxes

    @staticmethod
    def calculate_iou(box1, box2):
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (list): Coordinates of the bounding box in the format [x1, y1, x2, y2].
            box2 (list): Coordinates of the bounding box in the format [x1, y1, x2, y2].

        Returns:
            float: IoU value.
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

    @staticmethod
    def evaluate_bounding_boxes(pred_boxes, gt_boxes, iou_threshold):
        """
        Evaluate predicted bounding boxes with respect to ground truth using an IoU threshold.

        Args:
            pred_boxes (list): List of predicted bounding boxes.
            gt_boxes (list): List of ground truth bounding boxes.
            iou_threshold (float): IoU threshold to consider a detection as true positive.

        Returns:
            tuple: Values of TP, FN, FP.
        """
        TP = 0
        FN = 0
        FP = 0

        # Mark ground truth that have already been matched with a prediction
        matched_gt = [False] * len(gt_boxes)

        # Iterate over all predictions
        for pred_box in pred_boxes:
            max_iou = 0
            max_iou_index = -1

            # Find the ground truth with the highest IoU
            for i, gt_box in enumerate(gt_boxes):
                iou = Evaluator.calculate_iou(pred_box, gt_box)
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

        return TP, FN, FP



