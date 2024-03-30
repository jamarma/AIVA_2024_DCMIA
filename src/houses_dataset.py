import glob
import os
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.io import read_image
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
        image = read_image(self.images_paths[index])

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
        image = tv_tensors.Image(image)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image))
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        target["image_id"] = index

        # Applies transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
