import os
import glob
import cv2
import numpy as np
from natsort import natsorted
from sklearn.model_selection import train_test_split
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

TRAIN_DIR = '../data/train/'
TEST_DIR = '../data/test/'


def crop_patches(images_path: str, masks_path: str, patch_size: int) -> ([np.array], [np.array]):
    """
    Cuts the input images and masks into square patches of a given size.

    Parameters:
        - images_path (str): path to rgb images with .tif extension.
        - masks_path (str): path to binary masks with .tif extension.
        - patch_size (int): size of square patches.

    Returns:
        - img_patches ([np.array]): list containing the patches that
        was cut out of the images
        - mask_patches ([np.array]): list containing the patches that
        was cut out of the binary masks.
    """
    images = natsorted(glob.glob(f'{images_path}/*.tif'))
    masks = natsorted(glob.glob(f'{masks_path}/*.tif'))

    if len(images) != len(masks):
        raise ValueError('The number of images and masks does not match.')

    img_patches = []
    mask_patches = []
    for img_file, mask_file in zip(images, masks):
        img = cv2.imread(img_file)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        h, w, _ = img.shape

        for j in range(w // patch_size):
            for i in range(h // patch_size):
                x_init = i * patch_size
                y_init = j * patch_size
                x_end = x_init + patch_size
                y_end = y_init + patch_size

                mask_patch = np.copy(mask[y_init:y_end, x_init:x_end])
                img_patch = np.copy(img[y_init:y_end, x_init:x_end])
                img_patches.append(img_patch)
                mask_patches.append(mask_patch)
    return img_patches, mask_patches


def generate_pascal_voc_xml(filename: str, shape_image: (int, int, int), bounding_boxes: [(int, int, int, int)], classes: [str]):
    """
    Generates a Pascal VOC xml format string with the given bounding
    box and classes of the detections in an image.

    Parameters:
        - filename (str): name of the image file.
        - shape_image ((int, int, int)): shape of the imagen (height,
        width, channels)
        - bounding_boxes ([(int, int, int, int)]): list of bounding boxes
        (x, y, w, h) of objects in the image.
        - classes ([str]): list of classes of each bounding box.

    Returns:
        - xml_pretty_str (str): Pascal VOC xml.
    """
    annotation = Element('annotation')

    folder = SubElement(annotation, 'folder')
    folder.text = ''

    filename_elem = SubElement(annotation, 'filename')
    filename_elem.text = filename

    path = SubElement(annotation, 'path')
    path.text = filename

    source = SubElement(annotation, 'source')
    database = SubElement(source, 'database')
    database.text = ''

    size = SubElement(annotation, 'size')
    width_elem = SubElement(size, 'width')
    width_elem.text = str(shape_image[1])
    height_elem = SubElement(size, 'height')
    height_elem.text = str(shape_image[0])
    depth_elem = SubElement(size, 'depth')
    depth_elem.text = str(shape_image[2])

    segmented = SubElement(annotation, 'segmented')
    segmented.text = '0'

    for bbox, cls in zip(bounding_boxes, classes):
        obj = SubElement(annotation, 'object')

        name = SubElement(obj, 'name')
        name.text = cls

        pose = SubElement(obj, 'pose')
        pose.text = 'Unspecified'

        truncated = SubElement(obj, 'truncated')
        truncated.text = '0'

        difficult = SubElement(obj, 'difficult')
        difficult.text = '0'

        occluded = SubElement(obj, 'occluded')
        occluded.text = '0'

        bndbox = SubElement(obj, 'bndbox')
        xmin = SubElement(bndbox, 'xmin')
        xmin.text = str(bbox[0])
        xmax = SubElement(bndbox, 'xmax')
        xmax.text = str(bbox[0] + bbox[2])
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = str(bbox[1])
        ymax = SubElement(bndbox, 'ymax')
        ymax.text = str(bbox[1] + bbox[3])

    xml_string = tostring(annotation, 'utf-8')
    xml_pretty_str = parseString(xml_string).toprettyxml()

    return xml_pretty_str


def get_bounding_boxes(mask: np.array) -> ([(int, int, int, int)], [str]):
    """
    Returns the bounding boxes of the blobs of a binary image.

    - Parameters:
        - mask (np.array): the binary input image.

    - Returns:
        - bounding_boxes ([(int, int, int, int)]): list with the
        bounding boxes of the blobs with format (x, y, w, h).
        - classes ([str]): list with the classes of the blobs.
        All are 'house' class.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    classes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
        classes.append('house')
    return bounding_boxes, classes


def save_annotations_by_indices(img_patches: [np.array], mask_patches: [np.array], indices: [int], output_path: str):
    """
    Saves the patches in png format and the bounding boxes of each patch in
    Pascal VOC format in the output path provided. An index list is provided
    indicating which patches from the patch list should be saved.

    - Parameters:
        - img_patches (np.array): list of image patches.
        - mask_patches (np.array): list of mask patches.
        - indices ([int]): index list to select patches from the lists.
        - output_path (str): output path to save patches and annotations.
    """
    for idx in indices:
        img_patch = img_patches[idx]
        mask_patch = mask_patches[idx]
        bboxs, classes = get_bounding_boxes(mask_patch)
        filename = f'{idx}.png'
        xml = generate_pascal_voc_xml(filename, img_patch.shape, bboxs, classes)
        with open(os.path.join(output_path, f'{idx}.xml'), 'w') as f:
            f.write(xml)
        cv2.imwrite(os.path.join(output_path, filename), img_patch)


def save_patches_and_annotations(img_patches: [np.array], mask_patches: [np.array], test_size: float):
    """
    Divide the patches into training and test set. Save the patches in png
    format and the annotations (bounding box) of each patch in Pascal VOC
    format.

    - Parameters:
        - img_patches (np.array): list of image patches.
        - mask_patches (np.array): list of mask patches.
        - test_size (float): should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split.
    """
    train_indices, test_indices = train_test_split(range(len(img_patches)), test_size=test_size)

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    save_annotations_by_indices(img_patches, mask_patches, train_indices, TRAIN_DIR)
    save_annotations_by_indices(img_patches, mask_patches, test_indices, TEST_DIR)


img_patches, mask_patches = crop_patches('../data/raw/images',
                                         '../data/raw/masks',
                                         500)
save_patches_and_annotations(img_patches, mask_patches, test_size=0.2)


