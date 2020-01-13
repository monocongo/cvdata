import argparse
import logging
import os


# ------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def masks_from_vgg(
        images_dir: str,
        annotations: str,
        masks_dir: str,
):
    """
    TODO

    :param images_dir: directory containing image files to be filtered
    :param annotations_dir : annotation file containing segmentation (mask) regions,
        expected to be in the JSON format created by the VGG Annotator tool
    :param masks_dir: directory where mask files will be written
    """

    # arguments validation
    if not os.path.exists(images_dir):
        raise ValueError(f"Invalid images directory path: {images_dir}")
    elif not os.path.exists(annotations):
        raise ValueError(f"Invalid annotations file path: {annotations}")


# ------------------------------------------------------------------------------
def main():

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--images",
        required=True,
        type=str,
        help="path to directory containing input image files",
    )
    args_parser.add_argument(
        "--masks",
        required=True,
        type=str,
        help="path to directory where mask files should be written",
    )
    args_parser.add_argument(
        "--annotations",
        required=True,
        type=str,
        help="path to annotation file",
    )
    args_parser.add_argument(
        "--format",
        required=False,
        type=str,
        choices=["vgg", "coco", "openimages"],
        help="format of input annotations",
    )
    args = vars(args_parser.parse_args())

    if args["format"] == "vgg":
        masks_from_vgg(
            args["images"],
            args["annotations"],
            args["masks"],
        )


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Usage: 
    $ python mask.py --format vgg \
        --images /data/images \
        --annotations /data/via_annotations.json \
        --masks /data/masks
    """

    main()
