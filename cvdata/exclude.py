import argparse
import logging
import os
from typing import Set

import pandas as pd

from cvdata.common import FORMAT_CHOICES as format_choices


# ------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def exclude_files(
        exclusions_path: str,
        images_dir: str,
        annotations: str,
        annotation_format: str,
):
    """
    Removes all image and annotation files matching to the file IDs specified
    in an exclusions file.

    :param exclusions_path: absolute path to a file containing file IDs (one per
        line) to be used to match to files that should be excluded from a dataset
    :param images_dir: directory containing image files to be filtered
    :param annotations: directory containing image files to be filtered, or
        the absolute path to the annotations CSV if OpenImages format
    :param annotation_format: annotation format
    """

    # argument validation
    if annotation_format not in set(format_choices):
        raise ValueError(f"Unsupported annotation format: \'{annotation_format}\'")

    def remove_matching_files(
            removal_ids: Set[str],
            directory: str,
    ):
        """
        Removes files from the specified directory which have a base file name
        matching to any of the provided file IDs.

        :param removal_ids:
        :param directory:
        """
        for file_name in os.listdir(directory):
            file_id, _ = os.path.splitext(file_name)
            if file_id in removal_ids:
                os.remove(os.path.join(directory, file_name))

    # read the file IDs from the exclusions file
    with open(exclusions_path, "r") as exclusions_file:
        exclusion_ids = set([line.rstrip('\n') for line in exclusions_file])

    # remove any image files that are in the exclusions list
    remove_matching_files(exclusion_ids, images_dir)

    if annotation_format == "openimages":
        df = pd.read_csv(annotations)
        df = df[~df["ImageID"].isin(exclusion_ids)]
        df.to_csv(annotations)
    else:  # annotation file names match to image file names for other formats
        remove_matching_files(exclusion_ids, annotations)

        # TODO KITTI typically has associated text files listing the file IDs
        #   for training/testing etc. and as such should also be handled here


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Usage: 
    $ python exclude.py --format pascal \
        --annotations ~/datasets/handgun/annotations/pascal \
        --images ~/datasets/handgun/images \
        --exclusions ~/datasets/handgun/exclusions.txt
    """

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--exclusions",
        type=str,
        required=True,
        help="path to file containing file IDs (one per line) to exclude from "
             "the final dataset",
    )
    args_parser.add_argument(
        "--annotations",
        required=True,
        type=str,
        help="path to directory containing input annotation files to be "
             "converted, or for OpenImages format path to the CSV file",
    )
    args_parser.add_argument(
        "--images",
        required=True,
        type=str,
        help="path to directory containing input image files",
    )
    args_parser.add_argument(
        "--format",
        required=True,
        type=str,
        choices=format_choices,
        help="format of input annotations",
    )
    args = vars(args_parser.parse_args())

    exclude_files(
        args["exclusions"],
        args["images"],
        args["annotations"],
        args["format"],
    )
