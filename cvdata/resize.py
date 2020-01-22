import argparse
import concurrent.futures
import fileinput
import logging
import os
from typing import Dict
from xml.etree import ElementTree

import cv2
import numpy as np
from tqdm import tqdm

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
def resize_with_padding(
        image: np.ndarray,
        new_width: int,
        new_height: int,
) -> (np.ndarray, int, int):
    """
    Reads image data from a file and resizes it to the specified dimensions,
    preserving the aspect ratio and padding on the right and bottom as necessary.

    :param image: numpy array of image (pixel) data
    :param new_width:
    :param new_height:
    :return: resized image data and paddings (width and height)
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

    return padded_img, pad_right, pad_bottom


# ------------------------------------------------------------------------------
def _get_resized_image(
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

    padded_img, pad_right, pad_bottom = \
        resize_with_padding(image, new_width, new_height)

    # get the scaling factors that were used
    original_height, original_width = image.shape[:2]
    scale_x = (new_width - pad_right) / original_width
    scale_y = (new_height - pad_bottom) / original_height

    return padded_img, scale_x, scale_y


# ------------------------------------------------------------------------------
def _resize_image_label(arguments: Dict):
    """
    Wrapper function to be used for mapping to an iterable of arguments that
    will be passed to the resizing function.

    :param arguments:
    :return:
    """

    resize_image_label(
        arguments["file_id"],
        arguments["image_ext"],
        arguments["annotation_ext"],
        arguments["input_images_dir"],
        arguments["input_annotations_dir"],
        arguments["output_images_dir"],
        arguments["output_annotations_dir"],
        arguments["new_width"],
        arguments["new_height"],
        arguments["annotation_format"],
    )


# ------------------------------------------------------------------------------
def _resize_image(arguments: Dict):
    """
    Wrapper function to be used for mapping to an iterable of arguments that
    will be passed to the resizing function.

    :param arguments:
    :return:
    """

    resize_image(
        arguments["file_name"],
        arguments["input_images_dir"],
        arguments["output_images_dir"],
        arguments["new_width"],
        arguments["new_height"],
    )


# ------------------------------------------------------------------------------
def resize_image(
        file_name: str,
        input_images_dir: str,
        output_images_dir: str,
        new_width: int,
        new_height: int,
) -> int:
    """
    Resizes an image.

    :param file_name: file name of the image file
    :param input_images_dir: directory where image file is located
    :param output_images_dir: directory where the resized image file
        should be written
    :param new_width: new width to which the image should be resized
    :param new_height: new height to which the image should be resized
    :return: 0 to indicate successful completion
    """

    # read the image data into a numpy array and get the dimensions
    image_path = os.path.join(input_images_dir, file_name)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    original_height, original_width = image.shape[:2]

    # resize if necessary
    if (original_width != new_width) or (original_height != new_height):
        image, _, _ = _get_resized_image(image, new_width, new_height)

    # write the scaled/padded image to file in the output directory
    resized_image_path = os.path.join(output_images_dir, file_name)
    cv2.imwrite(resized_image_path, image)

    return 0


# ------------------------------------------------------------------------------
def resize_image_label(
        file_id: str,
        image_ext: str,
        annotation_ext: str,
        input_images_dir: str,
        input_annotations_dir: str,
        output_images_dir: str,
        output_annotations_dir: str,
        new_width: int,
        new_height: int,
        annotation_format: str,
) -> int:
    """
    Resizes an image and its corresponding annotation.

    :param file_id: file ID of the image and annotation files
    :param image_ext: file extension of the image file
    :param annotation_ext: file extension of the annotation file
    :param input_images_dir: directory where image file is located
    :param input_annotations_dir: directory where annotation file is located
    :param output_images_dir: directory where the resized image file
        should be written
    :param output_annotations_dir: directory where the resized annotation file
        should be written
    :param new_width: new width to which the image should be resized
    :param new_height: new height to which the image should be resized
    :param annotation_format: "coco", "darknet", "kitti", or "pascal"
    :return: 0 to indicate successful completion
    """

    # read the image data into a numpy array and get the dimensions
    image_file_name = file_id + image_ext
    image_path = os.path.join(input_images_dir, image_file_name)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    original_height, original_width = image.shape[:2]

    # resize if necessary
    if (original_width != new_width) or (original_height != new_height):
        image, scale_x, scale_y = _get_resized_image(image, new_width, new_height)
    else:
        scale_x = scale_y = 1.0

    # write the scaled/padded image to file in the output directory
    resized_image_path = os.path.join(output_images_dir, image_file_name)
    cv2.imwrite(resized_image_path, image)

    annotation_file_name = file_id + annotation_ext
    annotation_path = os.path.join(input_annotations_dir, annotation_file_name)

    if annotation_format == "pascal":

        tree = ElementTree.parse(annotation_path)
        root = tree.getroot()

        # update the image dimensions if they've changed
        if new_width != original_width:
            root.find("size").find("width").text = str(new_width)
        if new_height != original_height:
            root.find("size").find("height").text = str(new_height)

        # update any bounding boxes
        for bbox in root.iter("bndbox"):
            # get the min/max elements
            xmin = bbox.find("xmin")
            ymin = bbox.find("ymin")
            xmax = bbox.find("xmax")
            ymax = bbox.find("ymax")

            # clip to one less pixel than the dimension size in
            # case the scaling takes us all the way to the edge
            new_xmax = min((new_width - 1), int(int(xmax.text) * scale_x))
            new_ymax = min((new_height - 1), int(int(ymax.text) * scale_y))

            xmin.text = str(int(int(xmin.text) * scale_x))
            ymin.text = str(int(int(ymin.text) * scale_y))
            xmax.text = str(new_xmax)
            ymax.text = str(new_ymax)

        # update the image file path
        path = root.find("path")
        if path is not None:
            path.text = resized_image_path

        # write the updated XML into the annotations output directory
        tree.write(os.path.join(output_annotations_dir, annotation_file_name))

    elif annotation_format == "kitti":

        def scale_line(
                kitti_line: str,
                width_new: int,
                height_new: int,
                x_scale: float,
                y_scale: float,
        ) -> str:

            parts = kitti_line.rstrip("\r\n").split()
            x_min, y_min, x_max, y_max = list(map(int, map(float, parts[4:8])))

            # clip to one less pixel than the dimension size in
            # case the scaling takes us all the way to the edge
            xmin_new = str(int(x_min * x_scale))
            ymin_new = str(int(y_min * y_scale))
            xmax_new = str(min((width_new - 1), int(x_max * x_scale)))
            ymax_new = str(min((height_new - 1), int(y_max * y_scale)))

            parts[4:8] = xmin_new, ymin_new, xmax_new, ymax_new
            return " ".join(parts)

        output_annotation_path = \
            os.path.join(output_annotations_dir, annotation_file_name)

        if annotation_path == output_annotation_path:
            # replace the bounding boxes in-place
            with fileinput.FileInput(annotation_path, inplace=True) as file_input:
                for line in file_input:
                    print(scale_line(line, new_width, new_height, scale_x, scale_y))

        else:
            # read lines from the original, update the bounding box, and write to new file
            with open(annotation_path, "r") as original_kitti_file, \
                    open(output_annotation_path, "w") as new_kitti_file:
                for line in original_kitti_file:
                    new_kitti_file.write(scale_line(line, new_width, new_height, scale_x, scale_y))

    else:
        raise ValueError(f"Unsupported annotation format: \'{annotation_format}\'")

    return 0


# ------------------------------------------------------------------------------
def resize_dataset(
        input_images_dir: str,
        input_annotations_dir: str,
        output_images_dir: str,
        output_annotations_dir: str,
        new_width: int,
        new_height: int,
        annotation_format: str,
):
    """
    Resizes all images and corresponding annotations located within the
    specified directories.

    :param input_images_dir: directory where image files are located
    :param input_annotations_dir: directory where annotation files are located
    :param output_images_dir: directory where resized image files should be written
    :param output_annotations_dir: directory where resized annotation files
        should be written
    :param new_width: new width to which the image should be resized
    :param new_height: new height to which the image should be resized
    :param annotation_format: "coco", "darknet", "kitti", or "pascal"
    :return: the number of resized image/annotation files
    """

    # only allow for KITTI and PASCAL annotation formats
    if annotation_format not in ["kitti", "pascal"]:
        raise ValueError(f"Unsupported annotation format: {annotation_format}")
    else:
        annotation_ext = cvdata.common.FORMAT_EXTENSIONS[annotation_format]

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
        list(tqdm(executor.map(_resize_image_label, resize_arguments_list),
                  total=len(resize_arguments_list)))


# ------------------------------------------------------------------------------
def resize_images(
        input_images_dir: str,
        output_images_dir: str,
        new_width: int,
        new_height: int,
):
    """
    Resizes all images and corresponding annotations located within the
    specified directories.

    :param input_images_dir: directory where image files are located
    :param output_images_dir: directory where resized image files should be written
    :param new_width: new width to which the image should be resized
    :param new_height: new height to which the image should be resized
    :return: the number of resized image/annotation files
    """

    # create the destination directories in case they don't already exist
    os.makedirs(output_images_dir, exist_ok=True)

    resize_arguments_list = []
    for file_name in os.listdir(input_images_dir):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            resize_arguments = {
                "file_name": file_name,
                "input_images_dir": input_images_dir,
                "output_images_dir": output_images_dir,
                "new_width": new_width,
                "new_height": new_height,
            }
            resize_arguments_list.append(resize_arguments)

    # use a ProcessPoolExecutor to download the images in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:

        _logger.info("Resizing files")

        # use the executor to map the download function to the iterable of arguments
        list(tqdm(executor.map(_resize_image, resize_arguments_list),
                  total=len(resize_arguments_list)))


# ------------------------------------------------------------------------------
def main():

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
        required=False,
        type=str,
        help="directory path of original annotations",
    )
    args_parser.add_argument(
        "--output_annotations",
        required=False,
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
        choices=cvdata.common.FORMAT_CHOICES,
        help="output format: KITTI, PASCAL, Darknet, TFRecord, or COCO",
    )
    args = vars(args_parser.parse_args())

    if args["input_annotations"] is None:

        # resize only images
        resize_images(
            args["input_images"],
            args["output_images"],
            args["width"],
            args["height"],
        )

    else:

        # resize images and modify corresponding annotation files accordingly
        resize_dataset(
            args["input_images"],
            args["input_annotations"],
            args["output_images"],
            args["output_annotations"],
            args["width"],
            args["height"],
            args["format"],
        )


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # Example usages:
    #
    # Resize all images in a specified directory:
    #
    # $ python resize.py --input_images /ssd_training/kitti/image_2 \
    #     --output_images /ssd_training/kitti/image_2 \
    #     --width 1024 --height 768
    #
    #
    # Resize images and update the corresponding annotations:
    #
    # $ python resize.py --input_images /ssd_training/kitti/image_2 \
    #     --input_annotations /ssd_training/kitti/label_2 \
    #     --output_images /ssd_training/kitti/image_2 \
    #     --output_annotations /ssd_training/kitti/label_2 \
    #     --width 1024 --height 768 --format kitti

    main()
