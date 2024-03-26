import glob
import os
import cv2
import torch
from torch.utils.data import Dataset
from natsort import natsorted
from xml.etree import ElementTree as et


class HousesDataset(Dataset):
    def __init__(self, dir_path: str, classes: [str], transforms=None):
        self.dir_path = dir_path
        self.classes = classes
        self.images_paths = natsorted(glob.glob(f'{dir_path}/*.png'))
        self.transforms = transforms

    def __len__(self) -> int:
        """
        Returns the total number of samples.

        Returns:
            - int: the number of samples.
        """
        return len(self.images_paths)

    def __getitem__(self, index: int) -> (torch.Tensor, dict):
        """
        Returns a sample of dataset.

        Parameters:
            - index (int): index of sample.

        Returns:
            - image (torch.Tensor): the image of sample. If self.transforms
            is None, the image type will be np.array.
            - target (dict): dict with bounding boxes, labels and other
            information.
        """
        # Reads image from path
        image = cv2.imread(self.images_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Image name and annotations file path
        image_name, _ = os.path.splitext(os.path.basename(self.images_paths[index]))
        annot_file_path = os.path.join(self.dir_path, f'{image_name}.xml')

        # Reads bounding box and labels from annotations file
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        # Prepares the output target
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        target["image_id"] = torch.tensor([index])

        # Applies transforms
        if self.transforms is not None:
            sample = self.transforms(image=image,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image, target
