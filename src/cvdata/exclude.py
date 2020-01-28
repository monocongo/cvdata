import argparse
import logging
import os
from typing import Set

import pandas as pd

from cvdata.common import FORMAT_CHOICES


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
        annotations: str = None,
        annotation_format: str = None,
):
    """
    Removes all image (and, optionally) annotation files matching to the file
    IDs specified in an exclusions file.

    :param exclusions_path: absolute path to a file containing file IDs (one per
        line) to be used to match to files that should be excluded from a dataset
    :param images_dir: directory containing image files to be filtered
    :param annotations: directory containing annotation files to be filtered,
        or, in the case where we're using OpenImages format, the absolute path
        to the OpenImages bounding box annotations CSV file
    :param annotation_format: annotation format
    """

    # arguments validation
    if not os.path.exists(images_dir):
        raise ValueError(f"Invalid images directory path: {images_dir}")
    if annotations is not None:
        if annotation_format is None:
            raise ValueError(
                "Missing the format argument which is "
                "necessary to determine the annotation type",
            )
        elif annotation_format not in FORMAT_CHOICES:
            raise ValueError(f"Unsupported annotation format: \'{annotation_format}\'")
        elif not os.path.exists(annotations):
            raise ValueError(f"Invalid annotations directory path: {annotations}")

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

    # remove any annotation files that are in the exclusions list
    if annotations is not None:

        if annotation_format == "openimages":

            # open the annotations CSV as a pandas DataFrame
            df = pd.read_csv(annotations)

            # remove all rows that contain image IDs that match
            df = df[~df["ImageID"].isin(exclusion_ids)]

            # write the cleaned DataFrame back into the CSV file
            df.to_csv(annotations)

        else:

            remove_matching_files(exclusion_ids, annotations)

            # TODO KITTI typically has associated text files listing the file IDs
            #   for training/testing etc. and as such should also be handled here


# ------------------------------------------------------------------------------
def main():

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--exclusions",
        type=str,
        required=True,
        help="path to file containing file IDs (one per line) to exclude "
             "(remove) from the dataset",
    )
    args_parser.add_argument(
        "--annotations",
        required=False,
        type=str,
        help="path to directory containing input annotation files to be "
             "removed, or in the case of OpenImages format, the path to the CSV file",
    )
    args_parser.add_argument(
        "--images",
        required=True,
        type=str,
        help="path to directory containing input image files",
    )
    args_parser.add_argument(
        "--format",
        required=False,
        type=str,
        choices=FORMAT_CHOICES,
        help="format of input annotations",
    )
    args = vars(args_parser.parse_args())

    exclude_files(
        args["exclusions"],
        args["images"],
        args["annotations"],
        args["format"],
    )


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Usage: 
    $ python exclude.py --format pascal \
        --annotations ~/datasets/handgun/annotations/pascal \
        --images ~/datasets/handgun/images \
        --exclusions ~/datasets/handgun/exclusions.txt
    """

    main()
