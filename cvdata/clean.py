import argparse
import fileinput
import logging
import os
import shutil
from typing import Dict

from lxml import etree
from PIL import Image

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


# ------------------------------------------------------------------------------
def clean_kitti(
        labels_dir: str,
        images_dir: str,
        rename_labels: Dict,
        problems_dir: str = None,
):

    # convert all PNG images to JPG, and remove the original PNG file
    for file_id in matching_ids(labels_dir, images_dir, ".txt", ".png"):
        png_file_path = os.path.join(images_dir, file_id + ".png")
        png_to_jpg(png_file_path, remove_png=True)

    # get a set of file IDs of the KITTI-format annotations and corresponding images
    file_ids = matching_ids(labels_dir, images_dir, ".txt", ".jpg")

    # make the problem files directory if necessary, in case it doesn't already exist
    if problems_dir is not None:
        os.makedirs(problems_dir, exist_ok=True)

    # remove files that aren't matches
    for dir in [labels_dir, images_dir]:
        for file_name in os.listdir(dir):
            # only filter out image and KITTI label files (this is
            # needed in case a subdirectory exists in the directory)
            if file_name.endswith(".txt") or file_name.endswith(".jpg"):
                if os.path.splitext(file_name)[0] not in file_ids:
                    unmatched_file = os.path.join(dir, file_name)
                    if problems_dir is not None:
                        shutil.move(unmatched_file, os.path.join(problems_dir, file_name))
                    else:
                        os.remove(unmatched_file)

    # loop over all the matching files and clean the KITTI annotations
    for file_id in file_ids:

        # get the image width and height
        jpg_file_name = file_id + ".jpg"
        image_file_path = os.path.join(images_dir, jpg_file_name)
        image = Image.open(image_file_path)
        img_width, img_height = image.size

        # update the image file name in the KITTI label file
        src_annotation_file_path = os.path.join(labels_dir, file_id + ".txt")
        for line in fileinput.input(src_annotation_file_path, inplace=True):

            parts = line.split()
            label = parts[0]
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

            if label in rename_labels:
                # update the label
                label = rename_labels[label]

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
        rename_labels: Dict,
        problems_dir: str = None,
):

    # convert all PNG images to JPG, and remove the original PNG file
    for file_id in matching_ids(pascal_dir, images_dir, ".xml", ".png"):
        png_file_path = os.path.join(images_dir, file_id + ".png")
        png_to_jpg(png_file_path, remove_png=True)

    # get a set of file IDs of the PASCAL VOC annotations and corresponding images
    file_ids = matching_ids(pascal_dir, images_dir, ".xml", ".jpg")

    # make the problem files directory if necessary, in case it doesn't already exist
    if problems_dir is not None:
        os.makedirs(problems_dir, exist_ok=True)

    # remove files that aren't matches
    for dir in [pascal_dir, images_dir]:
        for file_name in os.listdir(dir):
            # only filter out image and PASCAL files (this is needed
            # in case a subdirectory exists in the directory)
            if file_name.endswith(".xml") or file_name.endswith(".jpg"):
                if os.path.splitext(file_name)[0] not in file_ids:
                    unmatched_file = os.path.join(dir, file_name)
                    if problems_dir is not None:
                        shutil.move(unmatched_file, os.path.join(problems_dir, file_name))
                    else:
                        os.remove(unmatched_file)

    # loop over all the matching files and clean the PASCAL annotations
    for i, file_id in enumerate(file_ids):

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

                # rename all bounding box labels if specified in the rename dictionary
                name = obj.find("name")
                if name is None:
                    # drop the bounding box since it is useless with no label
                    parent = obj.getparent()
                    parent.remove(obj)
                elif name.text in rename_labels:
                    # update the label
                    name.text = rename_labels[name.text]

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
if __name__ == "__main__":

    # Usage:
    # $ python clean.py --format pascal \
    #       --annotations_dir /data/datasets/delivery_truck/pascal \
    #       --images_dir /data/datasets/delivery_truck/images \
    #       --rename_labels deivery:delivery

    # parse the command line arguments
    format_choices = ["coco", "darknet", "kitti", "openimages", "pascal"]
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
        required=True,
        type=str,
        help="path to directory where we should move problem files",
    )
    args_parser.add_argument(
        "--format",
        required=True,
        type=str,
        choices=format_choices,
        help="format of input annotations",
    )
    args_parser.add_argument(
        "--rename_labels",
        required=False,
        type=str,
        help="labels to be renamed, in format new:old (space separated)",
    )
    args = vars(args_parser.parse_args())

    if args["rename_labels"]:
        renames = {}
        for rename_labels in args["rename_labels"].split():
            from_label, to_label = rename_labels.split(":")
            renames[from_label] = to_label

    if args["format"] == "kitti":

        clean_kitti(
            args["annotations_dir"],
            args["images_dir"],
            renames,
            args["problems_dir"],
        )

    elif args["format"] == "pascal":

        clean_pascal(
            args["annotations_dir"],
            args["images_dir"],
            renames,
            args["problems_dir"],
        )

