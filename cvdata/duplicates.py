import argparse
import logging
import os
import shutil
from typing import List

import imagehash
from PIL import Image
from tqdm import tqdm


# ------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def remove_duplicates(
        images_dir: str,
        annotations_dir: str = None,
        duplicates_dir: str = None,
) -> List[str]:
    """
    TODO
    """

    # create the duplicates directory in case it doesn't yet exist
    if duplicates_dir is not None:
        os.makedirs(duplicates_dir, exist_ok=True)

    # build a set of image hashes and a list of IDs that are duplicates
    _logger.info("Building image hashes and identifying duplicates...")
    image_hashes = set()
    duplicate_ids = []
    for image_file_name in tqdm(os.listdir(images_dir)):

        # only process JPG images
        if not image_file_name.endswith(".jpg"):
            continue

        # get a hash of the image and add the image ID to the list of duplicates
        # if it's already been added, otherwise add it to the set of hashes
        image = Image.open(os.path.join(images_dir, image_file_name))
        image_id = os.path.splitext(image_file_name)[0]
        image_hash = imagehash.dhash(image)
        if image_hash in image_hashes:
            duplicate_ids.append(image_id)
        else:
            image_hashes.add(image_hash)
    _logger.info("Done")

    # move or remove the duplicates
    _logger.info("Moving/removing duplicate images...")
    duplicate_files = []
    for duplicate_id in tqdm(duplicate_ids):

        image_file_name = duplicate_id + ".jpg"
        image_path = os.path.join(images_dir, image_file_name)
        duplicate_files.append(image_path)

        # either move or delete the image file
        if duplicates_dir is None:
            os.remove(image_path)
        else:
            shutil.move(image_path, os.path.join(duplicates_dir, image_file_name))
    _logger.info("Done")

    # move/remove corresponding annotations, if specified
    if annotations_dir is not None:
        _logger.info("Moving/removing corresponding duplicate annotations...")
        for annotation_file_name in tqdm(os.listdir(annotations_dir)):
            if os.path.splitext(annotation_file_name)[0] in duplicate_ids:
                annotation_path = os.path.join(annotations_dir, annotation_file_name)
                if duplicates_dir is None:
                    os.remove(annotation_path)
                else:
                    shutil.move(
                        annotation_path,
                        os.path.join(duplicates_dir, annotation_file_name),
                    )
        _logger.info("Done")

    return duplicate_files


# ------------------------------------------------------------------------------
def main():

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
        duplicates_dir=args["dups_dir"],
    )


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # Usage:
    #
    # $ python duplicates.py --images_dir /data/trucks/ups/images \
    # >      --annotations_dir /data/trucks/ups/pascal \
    # >      --dups_dir /data/trucks/ups/dups

    main()
