import argparse
import os
import shutil
from typing import Dict

from cvdata.common import FORMAT_CHOICES, FORMAT_EXTENSIONS
from cvdata.utils import matching_ids


# ------------------------------------------------------------------------------
def filter_class_boxes(
        src_images_dir: str,
        src_annotations_dir: str,
        dest_images_dir: str,
        dest_annotations_dir: str,
        class_counts: Dict,
        annotation_format: str,
):
    """
    TODO

    :param src_images_dir:
    :param src_annotations_dir:
    :param dest_images_dir:
    :param dest_annotations_dir:
    :param class_counts:
    :param annotation_format:
    """

    # make sure we don't have the same directories for src and dest
    if src_images_dir == dest_images_dir:
        raise ValueError(
            "Source and destination image directories are "
            "the same, must be different",
        )
    if src_annotations_dir == dest_annotations_dir:
        raise ValueError(
            "Source and destination annotation directories are "
            "the same, must be different",
        )

    # determine the file extension to be used for annotations
    if annotation_format not in ["darknet", "kitti", "pascal"]:
        raise ValueError(f"Unsupported annotation format: {annotation_format}")
    else:
        annotation_ext = FORMAT_EXTENSIONS[annotation_format]
    image_ext = ".jpg"

    # make the destination directories, in case they don't already exist
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_annotations_dir, exist_ok=True)

    # get all file IDs for image/annotation file matches
    file_ids = \
        matching_ids(
            src_annotations_dir,
            src_images_dir,
            annotation_ext,
            image_ext,
        )

    # keep a count of the boxes we've processed for each image class type
    processed_class_counts = {k: 0 for k in class_counts.keys()}

    for file_id in file_ids:

        include_file = False
        remove_labels = []
        annotation_file_name = file_id + annotation_ext
        src_annotation_path = os.path.join(src_annotations_dir, annotation_file_name)
        box_counts = count_boxes(src_annotation_path, annotation_format)
        for label in box_counts.keys():
            if label in class_counts:
                processed_class_count = processed_class_counts[label]
                if processed_class_counts[label] < class_counts[label]:
                    include_file = True
                    processed_class_counts[label] = processed_class_count + class_counts[label]
            else:
                remove_labels.append(label)

        if include_file:
            dest_annotation_path = os.path.join(dest_annotations_dir, annotation_file_name)
            if len(remove_labels) > 0:
                # remove the unnecessary labels from the annotation
                # and write it into the destination directory
                write_with_removed_labels(src_annotation_path, dest_annotation_path, remove_labels, annotation_format)
            else:
                # copy te annotation file into the destination directory as-is
                shutil.copy(src_annotation_path, dest_annotation_path)

            # copy the source image file into the destination images directory
            image_file_name = file_id + image_ext
            src_image_path = os.path.join(src_images_dir, image_file_name)
            dest_image_path = os.path.join(dest_images_dir, image_file_name)
            shutil.copy(src_image_path, dest_image_path)


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # Usage: filter a dataset down to a collection fo image and annotation files
    #        so that it contains a specified number of annotations (bounding
    #        boxes) per image class type
    #
    # $ python filter.py --format kitti --src_images /data/original/images \
    # >   --src_annotations /data/original/kitti \
    # >   --dest_images /data/filtered/images \
    # >   --dest_annotations /data/filtered/kitti \
    # >   --boxes_per_class dog:5000 cat:5000 panda:5000

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--src_annotations",
        required=True,
        type=str,
        help="path to directory containing original dataset's annotation files",
    )
    args_parser.add_argument(
        "--src_images",
        required=True,
        type=str,
        help="path to directory containing original dataset's image files",
    )
    args_parser.add_argument(
        "--format",
        required=False,
        type=str,
        choices=FORMAT_CHOICES,
        help="format of the annotations",
    )
    args_parser.add_argument(
        "--start",
        required=False,
        type=int,
        default=0,
        help="initial number to use in the enumeration",
    )
    args_parser.add_argument(
        "--boxes_per_class",
        required=False,
        type=str,
        nargs="*",
        help="counts of boxes per image class type, in format class:count "
             "(multiple counts are space separated)",
    )
    args = vars(args_parser.parse_args())

    # make a dictionary of class labels mapped to their maximum box counts
    class_counts = {}
    for class_count in args["boxes_per_class"].split():
        label, count = class_count.split(":")
        class_counts[label] = count

    # filter the dataset by class/count
    filter_class_boxes(
        args["src_images"],
        args["src_annotations"],
        args["dest_images"],
        args["dest_annotations"],
        class_counts,
    )
