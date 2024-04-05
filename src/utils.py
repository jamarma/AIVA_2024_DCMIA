import torch
import numpy as np
import cv2
import os

from torchvision.transforms import v2 as T
from argparse import ArgumentParser

from constants import CLASSES


def get_transform() -> T.Compose:
    """
    Returns a composition of transformations.

    Returns:
        - transforms (T.Compose): Composition of transformations.
    """
    transforms = []
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def collate_fn(batch: list[tuple]) -> tuple:
    """
    Collates a batch of samples.

    Parameters:
        - batch (list[tuple]): List of samples.

    Returns:
        - batch (tuple): Tuple of samples.
    """
    return tuple(zip(*batch))


def draw_boxes(image: np.array, boxes: np.array, labels: np.array = None) -> np.array:
    """
    Draws bounding boxes and labels in input image.

    Parameters:
        - image (np.array): the input image.
        - boxes (np.array): the bounding boxes to draw.
        - labels (np.array): the labels to draw.

    Returns:
        - output (np.array): the output image with boxes and
        labels drawn.
    """
    output = image.copy()
    for box_num in range(boxes.shape[0]):
        box = boxes[box_num]
        cv2.rectangle(
            output,
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0), 2
        )
        if labels is not None:
            label = CLASSES[labels[box_num]]
            cv2.putText(
                output, label, (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
    return output


def is_valid_output_image(parser: ArgumentParser, arg: str):
    """
    Checks if the destination directory exists and
    if the output image name has the desired extension.

    Parameters:
        - parser (ArgumentParser): the argument parser.
        - arg (str): the path to check.

    Returns:
        - arg (str): the path checked.
    """
    image_dir = os.path.dirname(arg) or '.'
    if not os.path.exists(image_dir):
        parser.error(f'The directory {image_dir} does not exist!')
    try:
        filename, extension = os.path.basename(arg).split('.')
    except:
        parser.error('Invalid file name of the output image')
    if extension != "png":
        parser.error('The file extension must be .png')
    return arg
