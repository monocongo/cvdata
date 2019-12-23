import argparse
import logging
import os
from typing import List


# ------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


def remove_duplicates(
        images_dir: str,
        annotations_dir: str = None,
        duplicates_dir: str = None,
) -> List[str]:

    pass


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--images_dir",
        required=True,
        type=str,
        help="path to directory of images",
    )
    args_parser.add_argument(
        "--annotations_dir",
        required=False,
        type=str,
        help="path to directory of annotations",
    )
    args_parser.add_argument(
        "--dups_dir",
        required=False,
        type=str,
        help="directory into which to move duplicate images and annotations",
    )
    args = vars(args_parser.parse_args())

    remove_duplicates(
        args["images_dir"],
        annotations_dir=args["annotations_dir"],
        dups_dir=args["dups_idr"],
    )
