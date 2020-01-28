import argparse
import fileinput
import logging
import os
import shutil
from typing import Dict, List, Set

from lxml import etree
from PIL import Image
from tqdm import tqdm

from cvdata.common import FORMAT_CHOICES, FORMAT_EXTENSIONS
from cvdata.convert import png_to_jpg
from cvdata.utils import matching_ids


# ------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


def purge_non_matching(
        images_dir: str,
        annotations_dir: str,
        annotation_format: str,
        problems_dir: str = None,
) -> Set[str]:
    """
    TODO

    :param images_dir:
    :param annotations_dir:
    :param annotation_format:
    :param problems_dir:
    :return:
    """

    # determine the file extensions
    if annotation_format not in ["darknet", "kitti", "pascal"]:
        raise ValueError(f"Unsupported annotation format: {annotation_format}")
    else:
        annotation_ext = FORMAT_EXTENSIONS[annotation_format]
    image_ext = ".jpg"

    # make the problem files directory if necessary, in case it doesn't already exist
    if problems_dir is not None:
        os.makedirs(problems_dir, exist_ok=True)

    # remove files that aren't matches
    matching_file_ids = matching_ids(annotations_dir, images_dir, annotation_ext, image_ext)
    for directory in [annotations_dir, images_dir]:
        for file_name in os.listdir(directory):
            # only filter out image and Darknet annotation files (this is
            # needed in case a subdirectory exists in the directory)
            # and skip the file named "labels.txt"
            if file_name != "labels.txt" and \
                    (file_name.endswith(annotation_ext) or file_name.endswith(image_ext)):
                if os.path.splitext(file_name)[0] not in matching_file_ids:
                    unmatched_file = os.path.join(directory, file_name)
                    if problems_dir is not None:
                        shutil.move(unmatched_file, os.path.join(problems_dir, file_name))
                    else:
                        os.remove(unmatched_file)

    return matching_file_ids


# ------------------------------------------------------------------------------
def clean_darknet(
        darknet_dir: str,
        images_dir: str,
        label_replacements: Dict,
        label_removals: List[str] = None,
        problems_dir: str = None,
):
    """
    TODO

    :param darknet_dir:
    :param images_dir:
    :param label_replacements:
    :param label_removals:
    :param problems_dir:
    :return:
    """

    _logger.info("Cleaning dataset with Darknet annotations")

    # convert all PNG images to JPG, and remove the original PNG file
    for file_id in matching_ids(darknet_dir, images_dir, ".txt", ".png"):
        png_file_path = os.path.join(images_dir, file_id + ".png")
        png_to_jpg(png_file_path, remove_png=True)

    # get the set of file IDs of the Darknet-format annotations and corresponding images
    file_ids = purge_non_matching(images_dir, darknet_dir, "darknet", problems_dir)

    # loop over all the matching files and clean the Darknet annotations
    for file_id in tqdm(file_ids):

        # update the Darknet annotation file
        src_annotation_file_path = os.path.join(darknet_dir, file_id + ".txt")
        for line in fileinput.input(src_annotation_file_path, inplace=True):

            # get the bounding box label
            parts = line.split()
            label = parts[0]

            # skip rewriting this line if it's a label we want removed
            if (label_removals is not None) and (label in label_removals):
                continue

            # get the bounding box coordinates
            center_x = float(parts[1])
            center_y = float(parts[2])
            bbox_width = float(parts[3])
            bbox_height = float(parts[4])

            if (label_replacements is not None) and (label in label_replacements):
                # update the label
                label = label_replacements[label]

            # make sure we don't have wonky bounding box values
            # and if so we'll skip them
            if (center_x > 1.0) or (center_x < 0.0):
                # report the issue via log message
                _logger.warning(
                    "Bounding box center X is out of valid range -- skipping "
                    f"in Darknet annotation file {src_annotation_file_path}",
                )
                continue

            if (center_y > 1.0) or (center_y < 0.0):
                # report the issue via log message
                _logger.warning(
                    "Bounding box center Y is out of valid range -- skipping "
                    f"in Darknet annotation file {src_annotation_file_path}",
                )
                continue

            if (bbox_width > 1.0) or (bbox_width < 0.0):
                # report the issue via log message
                _logger.warning(
                    "Bounding box width is out of valid range -- skipping "
                    f"in Darknet annotation file {src_annotation_file_path}",
                )
                continue

            if (bbox_height > 1.0) or (bbox_height < 0.0):
                # report the issue via log message
                _logger.warning(
                    "Bounding box height is out of valid range -- skipping "
                    f"in Darknet annotation file {src_annotation_file_path}",
                )
                continue

            # write the line back into the file in-place
            darknet_parts = [
                label,
                f'{center_x:.4f}',
                f'{center_y:.4f}',
                f'{bbox_width:.4f}',
                f'{bbox_height:.4f}',
            ]
            print(" ".join(darknet_parts))


# ------------------------------------------------------------------------------
def clean_kitti(
        kitti_dir: str,
        images_dir: str,
        label_replacements: Dict = None,
        label_removals: List[str] = None,
        problems_dir: str = None,
):
    """
    TODO

    :param kitti_dir:
    :param images_dir:
    :param label_replacements:
    :param label_removals:
    :param problems_dir:
    :return:
    """

    _logger.info("Cleaning dataset with KITTI annotations")

    # convert all PNG images to JPG, and remove the original PNG file
    for file_id in matching_ids(kitti_dir, images_dir, ".txt", ".png"):
        png_file_path = os.path.join(images_dir, file_id + ".png")
        png_to_jpg(png_file_path, remove_png=True)

    # get the set of file IDs of the Darknet-format annotations and corresponding images
    file_ids = purge_non_matching(images_dir, kitti_dir, "kitti", problems_dir)

    # loop over all the matching files and clean the KITTI annotations
    for file_id in tqdm(file_ids):

        # get the image width and height
        jpg_file_name = file_id + ".jpg"
        image_file_path = os.path.join(images_dir, jpg_file_name)
        image = Image.open(image_file_path)
        img_width, img_height = image.size

        # update the image file name in the KITTI annotation file
        src_annotation_file_path = os.path.join(kitti_dir, file_id + ".txt")
        for line in fileinput.input(src_annotation_file_path, inplace=True):

            parts = line.split()
            label = parts[0]

            # skip rewriting this line if it's a label we want removed
            if (label_removals is not None) and (label in label_removals):
                continue

            truncated = parts[1]
            occluded = parts[2]
            alpha = parts[3]
            bbox_min_x = int(float(parts[4]))
            bbox_min_y = int(float(parts[5]))
            bbox_max_x = int(float(parts[6]))
            bbox_max_y = int(float(parts[7]))
            dim_x = parts[8]
            dim_y = parts[9]
            dim_z = parts[10]
            loc_x = parts[11]
            loc_y = parts[12]
            loc_z = parts[13]
            rotation_y = parts[14]
            # not all KITTI-formatted files have a score field
            if len(parts) == 16:
                score = parts[15]
            else:
                score = " "

            if (label_replacements is not None) and (label in label_replacements):
                # update the label
                label = label_replacements[label]

            # make sure we don't have wonky bounding box values
            # with mins > maxs, and if so we'll reverse them
            if bbox_min_x > bbox_max_x:
                # report the issue via log message
                _logger.warning(
                    "Bounding box minimum X is greater than the maximum X "
                    f"in KITTI annotation file {src_annotation_file_path}",
                )
                tmp_holder = bbox_min_x
                bbox_min_x = bbox_max_x
                bbox_max_x = tmp_holder

            if bbox_min_y > bbox_max_y:
                # report the issue via log message
                _logger.warning(
                    "Bounding box minimum Y is greater than the maximum Y "
                    f"in KITTI annotation file {src_annotation_file_path}",
                )
                tmp_holder = bbox_min_y
                bbox_min_y = bbox_max_y
                bbox_max_y = tmp_holder

            # perform sanity checks on max values
            if bbox_max_x >= img_width:
                # report the issue via log message
                _logger.warning(
                    "Bounding box maximum X is greater than width in KITTI "
                    f"annotation file {src_annotation_file_path}",
                )

                # fix the issue
                bbox_max_x = img_width - 1

            if bbox_max_y >= img_height:
                # report the issue via log message
                _logger.warning(
                    "Bounding box maximum Y is greater than height in KITTI "
                    f"annotation file {src_annotation_file_path}",
                )

                # fix the issue
                bbox_max_y = img_height - 1

            # write the line back into the file in-place
            kitti_parts = [
                label,
                truncated,
                occluded,
                alpha,
                f'{bbox_min_x:.1f}',
                f'{bbox_min_y:.1f}',
                f'{bbox_max_x:.1f}',
                f'{bbox_max_y:.1f}',
                dim_x,
                dim_y,
                dim_z,
                loc_x,
                loc_y,
                loc_z,
                rotation_y,
            ]
            if len(parts) == 16:
                kitti_parts.append(score)
            print(" ".join(kitti_parts))


# ------------------------------------------------------------------------------
def clean_pascal(
        pascal_dir: str,
        images_dir: str,
        label_replacements: Dict = None,
        label_removals: List[str] = None,
        problems_dir: str = None,
):
    """
    TODO

    :param pascal_dir:
    :param images_dir:
    :param label_replacements:
    :param label_removals:
    :param problems_dir:
    :return:
    """

    _logger.info("Cleaning dataset with PASCAL annotations")

    # convert all PNG images to JPG, and remove the original PNG file
    for file_id in matching_ids(pascal_dir, images_dir, ".xml", ".png"):
        png_file_path = os.path.join(images_dir, file_id + ".png")
        png_to_jpg(png_file_path, remove_png=True)

    # get the set of file IDs of the Darknet-format annotations and corresponding images
    file_ids = purge_non_matching(images_dir, pascal_dir, "pascal", problems_dir)

    # loop over all the matching files and clean the PASCAL annotations
    for i, file_id in tqdm(enumerate(file_ids)):

        # get the image width and height
        jpg_file_name = file_id + ".jpg"
        image_file_path = os.path.join(images_dir, jpg_file_name)
        image = Image.open(image_file_path)
        img_width, img_height = image.size

        # update the image file name in the PASCAL file
        src_annotation_file_path = os.path.join(pascal_dir, file_id + ".xml")
        if os.path.exists(src_annotation_file_path):
            tree = etree.parse(src_annotation_file_path)
            root = tree.getroot()

            size = tree.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)

            if (width != img_width) or (height != img_height):
                # something's amiss that we can't reasonably fix, remove files
                if problems_dir is not None:
                    for file_path in [src_annotation_file_path, image_file_path]:
                        dest_file_path = os.path.join(problems_dir, os.path.split(file_path)[1])
                        shutil.move(file_path, dest_file_path)
                else:
                    os.remove(src_annotation_file_path)
                    os.remove(image_file_path)
                continue

            # update the image file name
            file_name = root.find("filename")
            if (file_name is not None) and (file_name.text != jpg_file_name):
                file_name.text = jpg_file_name

            # loop over all bounding boxes
            for obj in root.iter("object"):

                # replace all bounding box labels if specified in the replacement dictionary
                name = obj.find("name")
                if (name is None) or ((label_removals is not None) and (name.text in label_removals)):
                    # drop the bounding box
                    parent = obj.getparent()
                    parent.remove(obj)
                    # move on, nothing more to do for this box
                    continue
                elif (label_replacements is not None) and (name.text in label_replacements):
                    # update the label
                    name.text = label_replacements[name.text]

                # for each bounding box make sure we have max
                # values that are one less than the width/height
                bbox = obj.find("bndbox")
                bbox_min_x = int(float(bbox.find("xmin").text))
                bbox_min_y = int(float(bbox.find("ymin").text))
                bbox_max_x = int(float(bbox.find("xmax").text))
                bbox_max_y = int(float(bbox.find("ymax").text))

                # make sure we don't have wonky values with mins > maxs
                if (bbox_min_x >= bbox_max_x) or (bbox_min_y >= bbox_max_y):
                    # drop the bounding box
                    _logger.warning(
                        "Dropping bounding box for object in file "
                        f"{src_annotation_file_path} due to invalid "
                        "min/max values",
                    )
                    parent = obj.getparent()
                    parent.remove(obj)

                else:
                    # make sure the max values don't go past the edge
                    if bbox_max_x >= img_width:
                        bbox.find("xmax").text = str(img_width - 1)
                    if bbox_max_y >= img_height:
                        bbox.find("ymax").text = str(img_height - 1)

            # drop the image path, it's not reliable
            path = root.find("path")
            if path is not None:
                parent = path.getparent()
                parent.remove(path)

            # drop the image folder, it's not reliable
            folder = root.find("folder")
            if folder is not None:
                parent = folder.getparent()
                parent.remove(folder)

            # write the tree back to file
            tree.write(src_annotation_file_path)


# ------------------------------------------------------------------------------
def main():

    # Usage:
    # $ python clean.py --format pascal \
    #       --annotations_dir /data/datasets/delivery_truck/pascal \
    #       --images_dir /data/datasets/delivery_truck/images \
    #       --replace_labels deivery:delivery

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--annotations_dir",
        required=True,
        type=str,
        help="path to directory containing annotation files to be cleaned",
    )
    args_parser.add_argument(
        "--images_dir",
        required=True,
        type=str,
        help="path to directory containing image files",
    )
    args_parser.add_argument(
        "--problems_dir",
        required=False,
        type=str,
        help="path to directory where we should move problem files",
    )
    args_parser.add_argument(
        "--format",
        required=True,
        type=str,
        choices=FORMAT_CHOICES,
        help="format of input annotations",
    )
    args_parser.add_argument(
        "--replace_labels",
        required=False,
        type=str,
        nargs="*",
        help="labels to be replaced, in format new:old (space separated)",
    )
    args_parser.add_argument(
        "--remove_labels",
        required=False,
        type=str,
        nargs="*",
        help="labels of bounding boxes to be removed",
    )
    args = vars(args_parser.parse_args())

    # make a dictionary of labels to be replaced, if provided
    replacements = None
    if args["replace_labels"]:
        replacements = {}
        for replace_labels in args["replace_labels"].split():
            from_label, to_label = replace_labels.split(":")
            replacements[from_label] = to_label

    # map the cleaner functions to their corresponding formats
    cleaners = {
        "darknet": clean_darknet,
        "kitti": clean_kitti,
        "pascal": clean_pascal,
    }

    # call the appropriate cleaner function for the annotation format
    if args["format"] not in cleaners:
        raise ValueError(f"Unsupported annotation format: {args['format']}")
    else:
        cleaners[args["format"]](
            args["annotations_dir"],
            args["images_dir"],
            replacements,
            args["remove_labels"],
            args["problems_dir"],
        )


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    main()
