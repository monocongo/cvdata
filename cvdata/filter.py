import argparse
from typing import Dict

from cvdata.common import FORMAT_CHOICES


# ------------------------------------------------------------------------------
def filter_class_boxes(
        src_images_dir: str,
        src_annotations_dir: str,
        dest_images_dir: str,
        dest_annotations_dir: str,
        class_counts: Dict,
        format: str,
):
    """
    TODO

    :param src_images_dir:
    :param src_annotations_dir:
    :param dest_images_dir:
    :param dest_annotations_dir:
    :param class_counts:
    :param format:
    """

    # TODO
    pass


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
