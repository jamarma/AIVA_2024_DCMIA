import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from constants import (
    CLASSES
)


def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.Resize(500, 500),
        A.RandomRotate90(0.5),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def get_test_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def visualize_dataset_sample(image, target):
    image_arr = image.permute(1, 2, 0).numpy()
    for box_num in range(len(target['boxes'])):
        box = target['boxes'][box_num]
        label = CLASSES[target['labels'][box_num]]
        cv2.rectangle(
            image_arr,
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0), 2
        )
        cv2.putText(
            image_arr, label, (int(box[0]), int(box[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
    cv2.imshow('Image', image_arr)
    cv2.waitKey(0)
