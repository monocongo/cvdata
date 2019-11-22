import argparse
import json
import logging
import os
from typing import Dict
from xml.etree import ElementTree

import cv2
import pandas as pd

import cvdata.common
from cvdata.utils import matching_ids


# ------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def labels_count_coco(
        file_path: str,
) -> Dict:
    """
    TODO

    :param file_path: path to a COCO annotation file
    :return: dictionary object with label names as keys mapped to the count of
        how many bounding boxes for the label are present in the annotation file
    """

    with open(file_path) as json_file:
        data = json.load(json_file)

    counts = {}
    categories = {category["id"]: category["name"] for category in data["categories"]}
    for annotation in data["annotations"]:

        if annotation["category_id"] in categories:
            name = categories["category_id"]
            if name in counts:
                counts[name] += 1
            else:
                counts[name] = 1

    return counts


# ------------------------------------------------------------------------------
def labels_count_pascal(
        file_path: str,
) -> Dict:
    """
    Returns the count of labels in a PASCAL VOC annotation (*.xml) file.

    :param file_path: path to a PASCAL VOC annotation file
    :return: dictionary object with label names as keys mapped to the count of
        how many bounding boxes for the label are present in the annotation file
    """

    counts = {}

    # load the contents of the annotations file into an ElementTree
    tree = ElementTree.parse(file_path)

    for obj in tree.iter("object"):

        bbox_label = obj.find("name").text
        if bbox_label in counts:
            counts[bbox_label] += 1
        else:
            counts[bbox_label] = 1

    return counts


# ------------------------------------------------------------------------------
def labels_count_text(
        file_path: str,
) -> Dict:
    """
    TODO

    :param file_path: path to a Darknet annotation file
    :return: dictionary object with label names as keys mapped to the count of
        how many bounding boxes for the label are present in the annotation file
    """

    counts = {}
    with open(file_path) as txt_file:

        for line in txt_file.readlines():

            name = line.split()[0]

            if name in counts:
                counts[name] += 1
            else:
                counts[name] = 1

    return counts


# ------------------------------------------------------------------------------
def labels_count_tfrecord(
        file_path: str,
) -> Dict:
    """
    TODO

    :param file_path: path to a Darknet annotation file
    :return: dictionary object with label names as keys mapped to the count of
        how many bounding boxes for the label are present in the annotation file
    """

    raise ValueError("Unsupported format: \"tfrecord\"")


# ------------------------------------------------------------------------------
def count_labels(
        annotation_file_path: str,
        annotation_format: str,
) -> Dict:

    if annotation_format == "coco":
        return labels_count_coco(annotation_file_path)
    elif annotation_format in ["darknet", "kitti"]:
        return labels_count_text(annotation_file_path)
    elif annotation_format == "pascal":
        return labels_count_pascal(annotation_file_path)
    elif annotation_format == "tfrecord":
        return labels_count_tfrecord(annotation_file_path)
    else:
        raise ValueError(f"Unsupported annotation format: \"{annotation_format}\"")


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # Analyze the dataset comprised of a collection of images and corresponding
    # annotations.
    #
    # Usage:
    #
    # $ python analyze.py --images /home/ubuntu/data/handgun/images \
    #       --annotations /home/ubuntu/data/handgun/annotations/coco \
    #       --format coco
    #

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--annotations",
        required=True,
        type=str,
        help="annotations directory path",
    )
    args_parser.add_argument(
        "--images",
        required=True,
        type=str,
        help="images directory path",
    )
    args_parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=cvdata.common.FORMAT_CHOICES,
        help="annotation format",
    )
    args = vars(args_parser.parse_args())

    if args["format"] == "openimages":

        # read the OpenImages CSV into a pandas DataFrame
        df_annotations = pd.read_csv(args["annotations"])
        df_annotations = df_annotations[['ImageID', 'LabelName']]

        # TODO get another dataframe fromthe class descriptions and get the
        #  readable label names from there to map to the LabelName column

        # whittle it down to only the rows that match to image IDs
        file_ids = [os.path.splitext(file_name)[0] for file_name in os.listdir(args["images"])]
        df_annotations = df_annotations[df_annotations["ImageID"].isin(file_ids)]

        pass

    else:

        annotation_ext = cvdata.common.FORMAT_EXTENSIONS[args["format"]]

        # only annotations matching to the images are considered to be valid
        file_ids = matching_ids(args["annotations"], args["images"], annotation_ext, ".jpg")

        label_counts = {}
        for file_id in file_ids:
            annotation_file_path = \
                os.path.join(args["annotations"], file_id + annotation_ext)
            for label, count in count_labels(annotation_file_path, args["format"]).items():
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
