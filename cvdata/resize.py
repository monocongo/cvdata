import argparse
import concurrent.futures
import logging
import os
from typing import Dict
from xml.etree import ElementTree

import cv2
import numpy as np
from tqdm import tqdm

from cvdata.utils import matching_ids

_FORMAT_EXTENSIONS = {
    "coco": ".json",
    "darknet": ".txt",
    "kitti": ".txt",
    "pascal": ".xml",
}

# ------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def get_resized_image(
        image: np.ndarray,
        new_width: int,
        new_height: int,
) -> (np.ndarray, float, float):
    """
    Reads image data from a file and resizes it to the specified dimensions,
    preserving the aspect ratio and padding on the right and bottom as necessary.

    :param image: numpy array of image (pixel) data
    :param new_width:
    :param new_height:
    :return: resized image data and scale factors (width and height)
    """

    # get the dimensions and aspect ratio
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height

    # determine the interpolation method we'll use
    if (original_height > new_height) or (original_width > new_width):
        # use a shrinking algorithm for interpolation
        interp = cv2.INTER_AREA
    else:
        # use a stretching algorithm for interpolation
        interp = cv2.INTER_CUBIC

    # determine the new width and height (may differ from the width and
    # height arguments if using those doesn't preserve the aspect ratio)
    final_width = new_width
    final_height = round(final_width / aspect_ratio)
    if final_height > new_height:
        final_height = new_height
    final_width = round(final_height * aspect_ratio)

    # at this point we may be off by a few pixels, i.e. over
    # the specified new width or height values, so we'll clip
    # in order not to exceed the specified new dimensions
    final_width = min(final_width, new_width)
    final_height = min(final_height, new_height)

    # get the padding necessary to preserve aspect ratio
    pad_bottom = abs(new_height - final_height)
    pad_right = abs(new_width - final_width)

    # scale and pad the image
    scaled_img = cv2.resize(image, (final_width, final_height), interpolation=interp)
    padded_img = cv2.copyMakeBorder(
        scaled_img, 0, pad_bottom, 0, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0],
    )

    # get the scaling factors that were used
    scale_x = (new_width - pad_right) / original_width
    scale_y = (new_height - pad_bottom) / original_height

    return padded_img, scale_x, scale_y


# ------------------------------------------------------------------------------
def resize_image(arguments: Dict):

    image_file_name = arguments["file_id"] + arguments["image_ext"]
    image_path = os.path.join(arguments["input_images_dir"], image_file_name)

    # read the image data into a numpy array and get the dimensions
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    original_height, original_width = image.shape[:2]

    # resize if necessary
    if (original_width != arguments["new_width"]) or \
            (original_height != arguments["new_height"]):
        image, scale_x, scale_y = \
            get_resized_image(
                image,
                arguments["new_width"],
                arguments["new_height"],
            )
    else:
        scale_x = scale_y = 1.0

    # write the scaled/padded image to file in the output directory
    resized_image_path = os.path.join(arguments["output_images_dir"], image_file_name)
    cv2.imwrite(resized_image_path, image)

    annotation_file_name = arguments["file_id"] + arguments["annotation_ext"]
    annotation_path = os.path.join(arguments["input_annotations_dir"], annotation_file_name)

    if arguments["annotation_format"] == "pascal":

        tree = ElementTree.parse(annotation_path)
        root = tree.getroot()

        # update the image dimensions if they've changed
        if arguments["new_width"] != original_width:
            root.find("size").find("width").text = str(arguments["new_width"])
        if arguments["new_height"] != original_height:
            root.find("size").find("height").text = str(arguments["new_height"])

        # update any bounding boxes
        for bbox in root.iter("bndbox"):
            # get the min/max elements
            xmin = bbox.find("xmin")
            ymin = bbox.find("ymin")
            xmax = bbox.find("xmax")
            ymax = bbox.find("ymax")

            # clip to one less pixel than the dimension size in
            # case the scaling takes us all the way to the edge
            new_xmax = min((arguments["new_width"] - 1), int(int(xmax.text) * scale_x))
            new_ymax = min((arguments["new_height"] - 1), int(int(ymax.text) * scale_y))

            xmin.text = str(int(int(xmin.text) * scale_x))
            ymin.text = str(int(int(ymin.text) * scale_y))
            xmax.text = str(new_xmax)
            ymax.text = str(new_ymax)

        # update the image file path
        path = root.find("path")
        if path is not None:
            path.text = resized_image_path

        # write the updated XML into the annotations output directory
        tree.write(os.path.join(arguments["output_annotations_dir"], annotation_file_name))


# ------------------------------------------------------------------------------
def resize_images(
        input_images_dir: str,
        input_annotations_dir: str,
        output_images_dir: str,
        output_annotations_dir: str,
        new_width: int,
        new_height: int,
        annotation_format: str,
):
    """
    TODO
    :param input_images_dir:
    :param input_annotations_dir:
    :param output_images_dir:
    :param output_annotations_dir:
    :param new_width:
    :param new_height:
    :param annotation_format:
    :return: the number of resized image/annotation files
    """

    # only allow for PASCAL format
    if annotation_format != "pascal":
        raise ValueError(f"Unsupported annotation format: {annotation_format}")
    else:
        annotation_ext = _FORMAT_EXTENSIONS[annotation_format]

    # create the destination directories in case they don't already exist
    os.makedirs(output_annotations_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)

    resize_arguments_list = []
    for image_ext in (".jpg", ".png"):

        # find matching annotation and image files (i.e. same base file name)
        file_ids = matching_ids(
            input_annotations_dir,
            input_images_dir,
            annotation_ext,
            image_ext,
        )

        # loop over all image files and perform scaling/padding on each
        for file_id in file_ids:

            resize_arguments = {
                "file_id": file_id,
                "image_ext": image_ext,
                "annotation_format": annotation_format,
                "annotation_ext": annotation_ext,
                "input_images_dir": input_images_dir,
                "input_annotations_dir": input_annotations_dir,
                "output_images_dir": output_images_dir,
                "output_annotations_dir": output_annotations_dir,
                "new_width": new_width,
                "new_height": new_height,
            }
            resize_arguments_list.append(resize_arguments)

    # use a ProcessPoolExecutor to download the images in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:

        _logger.info("Resizing files")

        # use the executor to map the download function to the iterable of arguments
        list(tqdm(executor.map(resize_image, resize_arguments_list),
                  total=len(resize_arguments_list)))


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--input_images",
        required=True,
        type=str,
        help="directory path of original images",
    )
    args_parser.add_argument(
        "--output_images",
        required=True,
        type=str,
        help="directory path of resized images",
    )
    args_parser.add_argument(
        "--input_annotations",
        required=True,
        type=str,
        help="directory path of original annotations",
    )
    args_parser.add_argument(
        "--output_annotations",
        required=True,
        type=str,
        help="directory path of resized annotations",
    )
    args_parser.add_argument(
        "--width",
        required=True,
        type=int,
        help="width of resized images",
    )
    args_parser.add_argument(
        "--height",
        required=True,
        type=int,
        help="height of resized images",
    )
    args_parser.add_argument(
        "--format",
        type=str,
        required=False,
        default="pascal",
        choices=["pascal"],
        # choices=["darknet", "coco", "kitti", "pascal", "tfrecord"],
        help="output format: KITTI, PASCAL, Darknet, TFRecord, or COCO",
    )
    args = vars(args_parser.parse_args())

    resize_images(
        args["input_images"],
        args["input_annotations"],
        args["output_images"],
        args["output_annotations"],
        args["width"],
        args["height"],
        args["format"],
    )
