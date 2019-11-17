import argparse
import json
import logging
import os
from typing import List
from xml.etree import ElementTree

import cv2

# ------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def bbox_coco(
        file_path: str,
) -> List[dict]:
    """
    Returns the labeled bounding boxes from a COCO annotation (*.json) file.

    :param file_path: path to a COCO annotation file
    :return: list of bounding box dictionary objects with keys "label", "x",
        "y", "w", and "h"
    """
    with open(file_path) as json_file:
        data = json.load(json_file)

    boxes = []
    for annotation in data["annotations"]:

        x, y, w, h = annotation["bbox"]
        box = {"label": "", "x": x, "y": y, "w": w, "h": h}
        boxes.append(box)

    return boxes


# ------------------------------------------------------------------------------
def bbox_darknet(
        file_path: str,
        width: int,
        height: int,
) -> List[dict]:
    """
    Returns the labeled bounding boxes from a Darknet annotation (*.txt) file.

    :param file_path: path to a Darknet annotation file
    :param width: width of the corresponding image
    :param height: height of the corresponding image
    :return: list of bounding box dictionary objects with keys "label", "x",
        "y", "w", and "h"
    """

    boxes = []
    with open(file_path) as txt_file:

        for line in txt_file.readlines():
            parts = line.split()
            bbox_width = int(float(parts[3]) * width)
            bbox_height = int(float(parts[4]) * height)
            box = {
                # denormalize the bounding box coordinates
                "label": parts[0],
                "x": int(float(parts[1]) * width) - int(bbox_width / 2),
                "y": int(float(parts[2]) * height) - int(bbox_height / 2),
                "w": bbox_width,
                "h": bbox_height,
            }
            boxes.append(box)

    return boxes


# ------------------------------------------------------------------------------
def bbox_kitti(
        file_path: str,
        width: int,
        height: int,
) -> List[dict]:
    """
    Returns the labeled bounding boxes from a KITTI annotation (*.txt) file.

    :param file_path: path to a KITTI annotation file
    :param width: width of the corresponding image
    :param height: height of the corresponding image
    :return: list of bounding box dictionary objects with keys "label", "x",
        "y", "w", and "h"
    """

    boxes = []
    with open(file_path) as txt_file:

        for line in txt_file.readlines():

            parts = line.split()
            label = parts[0]
            bbox_min_x = int(float(parts[4]))
            bbox_min_y = int(float(parts[5]))
            bbox_max_x = int(float(parts[6]))
            bbox_max_y = int(float(parts[7]))

            # perform sanity checks on max values, failures won't cause a skip
            if bbox_max_x >= width:
                # report the issue via log message
                _logger.warning(
                    "Bounding box maximum X is greater than width in KITTI "
                    f"annotation file {file_path}",
                )

                # fix the issue
                bbox_max_x = width - 1

            if bbox_max_y >= height:
                # report the issue via log message
                _logger.warning(
                    "Bounding box maximum Y is greater than height in KITTI "
                    f"annotation file {file_path}",
                )

                # fix the issue
                bbox_max_y = height - 1

            # get the box's dimensions
            bbox_width = bbox_max_x - bbox_min_x + 1
            bbox_height = bbox_max_y - bbox_min_y + 1

            # make sure we don't have wonky values with mins > maxs
            if (bbox_min_x >= bbox_max_x) or (bbox_min_y >= bbox_max_y):
                # report the issue via log message
                _logger.warning(
                    "Bounding box minimum(s) greater than maximum(s) in KITTI "
                    f"annotation file {file_path} -- skipping",
                )

                # skip this box since it's not clear how to fix it
                continue

            # include this box in the list we'll return
            box = {
                "label": label,
                "x": bbox_min_x,
                "y": bbox_min_y,
                "w": bbox_width,
                "h": bbox_height,
            }
            boxes.append(box)

    return boxes


# ------------------------------------------------------------------------------
def bbox_pascal(
        file_path: str,
        width: int,
        height: int,
) -> List[dict]:
    """
    Returns the labeled bounding boxes from a PASCAL VOC annotation (*.xml) file.

    :param file_path: path to a PASCAL VOC annotation file
    :param width: width of the corresponding image
    :param height: height of the corresponding image
    :return: list of bounding box dictionary objects with keys "label", "x",
        "y", "w", and "h"
    """

    boxes = []

    # load the contents of the annotations file into an ElementTree
    tree = ElementTree.parse(file_path)

    # sanity check on the expected image dimensions
    size = tree.find("size")
    expected_width = int(size.find("width").text)
    expected_height = int(size.find("height").text)
    if expected_width != width:
        raise ValueError(
            f"File: {file_path}\nUnexpected width of image -- "
            f"expected: {expected_width}, actual: {width}",
        )
    if expected_height != height:
        raise ValueError(
            f"File: {file_path}\nUnexpected height of image -- "
            f"expected: {expected_height}, actual: {height}",
        )

    for obj in tree.iter("object"):

        label = obj.find("name").text
        bndbox = obj.find("bndbox")
        bbox_min_x = int(float(bndbox.find("xmin").text))
        bbox_min_y = int(float(bndbox.find("ymin").text))
        bbox_max_x = int(float(bndbox.find("xmax").text))
        bbox_max_y = int(float(bndbox.find("ymax").text))

        # perform sanity checks on max values, failures won't cause a skip
        if bbox_max_x >= width:
            # report the issue via log message
            _logger.warning("Bounding box maximum X is greater than width")

            # fix the issue
            bbox_max_x = width - 1

        if bbox_max_y >= height:
            # report the issue via log message
            _logger.warning("Bounding box maximum Y is greater than height")

            # fix the issue
            bbox_max_y = height - 1

        # get the box's dimensions
        bbox_width = bbox_max_x - bbox_min_x + 1
        bbox_height = bbox_max_y - bbox_min_y + 1

        # make sure we don't have wonky values with mins > maxs
        if (bbox_min_x >= bbox_max_x) or (bbox_min_y >= bbox_max_y):

            # report the issue via log message
            _logger.warning("Bounding box minimum(s) greater than maximum(s)")

            # skip this box since it's not clear how to fix it
            continue

        # include this box in the list we'll return
        box = {
            "label": label,
            "x": bbox_min_x,
            "y": bbox_min_y,
            "w": bbox_width,
            "h": bbox_height,
        }
        boxes.append(box)

    return boxes


# ------------------------------------------------------------------------------
def get_image_file_name(pascal_file_path: str) -> str:

    tree = ElementTree.parse(pascal_file_path)
    root = tree.getroot()
    return root.find("filename").text


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # Visualize images with bounding boxes specified by corresponding annotations.
    #
    # Usage:
    #
    # $ python <this_script.py> --images_dir /home/ubuntu/data/handgun/images \
    #       --annotations_dir /home/ubuntu/data/handgun/annotations/coco \
    #       --format coco
    #

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--annotations_dir",
        required=True,
        type=str,
        help="annotations directory path",
    )
    args_parser.add_argument(
        "--images_dir",
        required=True,
        type=str,
        help="images directory path",
    )
    args_parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=["coco", "darknet", "kitti", "pascal"],
        help="annotation format",
    )
    args = vars(args_parser.parse_args())

    # for each annotation file we'll draw all bounding boxes on the corresponding image
    count = 0
    for annotation_file_name in os.listdir(args["annotations_dir"]):

        count += 1

        if args["format"] == "pascal":
            # get the corresponding image file name from the annotation file
            image_file_name = get_image_file_name(
                os.path.join(args["annotations_dir"], annotation_file_name),
            )
        else:
            # we assume each annotation shares the same base name as the corresponding image
            image_file_name = os.path.splitext(annotation_file_name)[0] + ".jpg"

        # load the input image from disk to determine the dimensions
        image_file_path = \
            os.path.join(args["images_dir"], image_file_name)
        image = cv2.imread(image_file_path)
        try:
            image_height, image_width = image.shape[:2]
        except AttributeError as attribute_error:
            _logger.error(f"Bad image: {image_file_path}", attribute_error)
            continue

        # read the bounding boxes from the annotation file
        bboxes = []
        annotations_file_path = \
            os.path.join(args["annotations_dir"], annotation_file_name)
        if args["format"] == "coco":
            if annotation_file_name.endswith(".json"):
                bboxes = bbox_coco(annotations_file_path)
            else:
                continue
        elif args["format"] == "darknet":
            if annotation_file_name.endswith(".txt"):
                bboxes = bbox_darknet(annotations_file_path, image_width, image_height)
            else:
                continue
        elif args["format"] == "kitti":
            if annotation_file_name.endswith(".txt"):
                bboxes = bbox_kitti(annotations_file_path, image_width, image_height)
            else:
                continue
        elif args["format"] == "pascal":
            if annotation_file_name.endswith(".xml"):
                try:
                    bboxes = bbox_pascal(annotations_file_path, image_width, image_height)
                except ValueError as value_error:
                    _logger.error(f"Bad annotation: {annotations_file_path}", value_error)
                    continue
            else:
                continue
        else:
            raise ValueError(f"Invalid format argument: \'{args['format']}\'")

        # for each bounding box draw the rectangle
        for bbox in bboxes:

            # draw the box
            cv2.rectangle(
                image,
                (bbox["x"], bbox["y"]),
                (bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]),
                (0, 255, 0),
                2,
            )
            # draw the label
            cv2.putText(
                image,
                bbox["label"],
                (bbox["x"], bbox["y"]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (123, 234, 56),
                1,
            )
        _logger.info(f"{count} Displaying {len(bboxes)} boxes for {image_file_name}")

        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
