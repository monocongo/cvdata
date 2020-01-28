import argparse
import os

from cvdata.common import FORMAT_CHOICES


# ------------------------------------------------------------------------------
def rename_image_files(
        images_dir: str,
        prefix: str,
        start: int,
        digits: int,
):
    """
    Renames all images in a directory to <prefix>_<enumeration>.<original_ext>,
    with the enumeration portion starting at a designated number and with a
    specified number of digits width.

    :param images_dir: all image files within this directory will be renamed
    :param prefix: the prefix used for the new file names
    :param start: the number at which the enumeration portion of the new file
        names should begin
    :param digits: the number of digits (width) of the enumeration portion of the
        new file names
    """

    supported_extensions = ("gif", "jpg", "jpeg", "png",)
    current = start
    for file_name in os.listdir(images_dir):
        _, ext = os.path.splitext(file_name)
        if ext[1:].lower() in supported_extensions:
            new_file_name = f"{prefix}_{str(current).zfill(digits)}{ext}"
            new_file_path = os.path.join(images_dir, new_file_name)
            original_file_path = os.path.join(images_dir, file_name)
            os.rename(original_file_path, new_file_path)
            current += 1


# ------------------------------------------------------------------------------
def main():

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--annotations_dir",
        required=False,
        type=str,
        help="path to directory containing input annotation files to be converted",
    )
    args_parser.add_argument(
        "--images_dir",
        required=True,
        type=str,
        help="path to directory containing input image files",
    )
    args_parser.add_argument(
        "--prefix",
        required=True,
        type=str,
        help="file name prefix",
    )
    args_parser.add_argument(
        "--kitti_ids_file",
        required=False,
        type=str,
        help="name of the file that contains the file IDs for a dataset with "
             "annotations in KITTI format",
    )
    args_parser.add_argument(
        "--digits",
        required=False,
        type=int,
        default=6,
        help="the number of digits in the enumeration portion of the resulting "
             "file names",
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
    args = vars(args_parser.parse_args())

    if args["annotations_dir"] is None:

        rename_image_files(
            args["images_dir"],
            args["prefix"],
            args["start"],
            args["digits"],
        )
    else:
        raise ValueError("Renaming of annotated datasets is unsupported")


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # Usage: rename names of dataset files (images and annotations)
    # $ python rename.py --annotations_dir ~/datasets/handgun/kitti \
    #     --images_dir ~/datasets/handgun/images \
    #     --prefix handgun --start 100 --digits 6 \
    #     --format kitti --kitti_ids_file file_ids.txt
    #
    # Usage: rename names of image files (images only)
    # $ python rename.py --images_dir ~/datasets/handgun/images \
    #     --prefix handgun --start 100 --digits 6

    main()
