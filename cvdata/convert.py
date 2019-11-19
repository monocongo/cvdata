import argparse
import concurrent.futures
import logging
import os
from pathlib import Path
import shutil
from typing import Dict
from xml.etree import ElementTree

import cv2
from tqdm import tqdm

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
def pascal_png_to_jpg(
        pascal_dir: str,
        images_dir: str,
):

    _logger.info(f"Converting from PNG to JPG for images in directory {images_dir}")

    for file_name in tqdm(os.listdir(images_dir)):
        file_id, extension = os.path.splitext(file_name)
        if extension[1:].strip().lower() == ".png":
            # convert the image from PNG to JPG
            jpg_file_path = os.path.join(images_dir, file_id + ".jpg")
            png_file_name = file_id + ".png"
            png_file_path = os.path.join(images_dir, png_file_name)
            img = cv2.imread(png_file_path)
            cv2.imwrite(jpg_file_path, img)

            # update the filename and path elements to reflect the new file type
            annotation_path = os.path.join(pascal_dir, file_id + ".xml")
            tree = ElementTree.parse(annotation_path)
            root = tree.getroot()
            file_name = root.find("filename")
            if file_name is not None:
                file_name.text = png_file_name
            path = root.find("path")
            if path is not None:
                path.text = png_file_path
            tree.write(annotation_path)


# ------------------------------------------------------------------------------
def single_pascal_to_kitti(arguments: Dict):

    pascal_file_name = arguments["file_id"] + arguments["pascal_ext"]
    image_file_name = arguments["file_id"] + arguments["img_ext"]
    pascal_file_path = os.path.join(arguments["pascal_dir"], pascal_file_name)
    image_file_path = os.path.join(arguments["images_dir"], image_file_name)
    kitti_file_path = os.path.join(arguments["kitti_labels_dir"], arguments["file_id"] + ".txt")

    with open(kitti_file_path, "w") as kitti_file:

        tree = ElementTree.parse(pascal_file_path)
        root = tree.getroot()
        for obj in root.iter("object"):

            name = obj.find("name")
            if name is None:
                name = "unknown"
            else:
                name = name.text

            truncation = obj.find("truncated")
            if truncation is None:
                truncation = 0.0
            else:
                truncation = float(truncation.text)

            # get the min/max elements
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            kitti_file.write(
                f"{name} {truncation:.1f} 0 0.0 {xmin:.1f} {ymin:.1f} "
                f"{xmax:.1f} {ymax:.1f} 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n",
            )

    # put the image file into the KITTI images directory
    if arguments["move_image_files"]:
        shutil.move(image_file_path, arguments["kitti_images_dir"])
    else:
        shutil.copy2(image_file_path, arguments["kitti_images_dir"])


# ------------------------------------------------------------------------------
def pascal_to_kitti(
        pascal_dir: str,
        images_dir: str,
        kitti_data_dir: str,
        kitti_ids_file_name: str,
        move_image_files: bool = False,
) -> int:
    """
    Builds KITTI annotation files from PASCAL VOC annotation XML files.

    :param pascal_dir: directory containing input PASCAL VOC annotation
        XML files, all XML files in this directory matching to corresponding JPG
        files in the images directory will be converted to KITTI format
    :param images_dir: directory containing image files corresponding to the
        PASCAL VOC annotation files to be converted to KITTI format, all files
        matching to PASCAL VOC annotation files will be either copied (default)
        or moved (if move_image_files is True) to <kitti_data_dir>/image_2
    :param kitti_data_dir: directory under which images will be copied or moved
        into a subdirectory named "image_2" and KITTI annotation files will be
        written into a subdirectory named "label_2"
    :param kitti_ids_file_name: file name to contain all file IDs, to be written
        into the parent directory above the <kitti_data_dir>
    :param move_image_files: whether or not to move image files to
        <kitti_data_dir>/image_2 (default is False and image files are copied
        instead)
    :return:
    """

    _logger.info(f"Converting from PASCAL to KITTI for images in directory {images_dir}")

    # create the KITTI directories in case they don't already exist
    kitti_images_dir = os.path.join(kitti_data_dir, "image_2")
    kitti_labels_dir = os.path.join(kitti_data_dir, "label_2")
    for data_dir_name in (kitti_images_dir, kitti_labels_dir):
        os.makedirs(os.path.join(kitti_data_dir, data_dir_name), exist_ok=True)

    # assumed file extensions
    img_ext = ".jpg"
    pascal_ext = ".xml"

    # get list of file IDs of the PASCAL VOC annotations and corresponding images
    file_ids = matching_ids(pascal_dir, images_dir, pascal_ext, img_ext)

    # write the KITTI file IDs file in the KITTI data directory's parent directory
    kitti_ids_file_path = os.path.join(Path(kitti_data_dir).parent, kitti_ids_file_name)
    with open(kitti_ids_file_path, "w") as kitti_ids_file:
        for file_id in file_ids:
            kitti_ids_file.write(f"{file_id}\n")

    # build KITTI annotations from PASCAL and copy or
    # move the image files into KITTI images directory
    conversion_arguments_list = []
    for file_id in file_ids:

        conversion_arguments = {
            "file_id": file_id,
            "pascal_ext": pascal_ext,
            "img_ext": img_ext,
            "pascal_dir": pascal_dir,
            "images_dir": images_dir,
            "kitti_labels_dir": kitti_labels_dir,
            "kitti_images_dir": kitti_images_dir,
            "move_image_files": move_image_files,
        }
        conversion_arguments_list.append(conversion_arguments)

    # use a ProcessPoolExecutor to download the images in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:

        # use the executor to map the download function to the iterable of arguments
        _logger.info(f"Building KITTI labels in directory {kitti_labels_dir} ")
        list(tqdm(executor.map(single_pascal_to_kitti, conversion_arguments_list),
                  total=len(conversion_arguments_list)))

    # return the number of annotations converted
    return len(file_ids)


# ------------------------------------------------------------------------------
def pascal_to_openimages(
        pascal_dir: str,
        images_dir: str,
        openimages_dir:str,
):

    file_ids = matching_ids(pascal_dir, images_dir)


# ------------------------------------------------------------------------------
def openimages_to_kitti():
    # TODO
    pass


# ------------------------------------------------------------------------------
def openimages_png_to_jpg():
    # TODO
    pass


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Usage: PASCAL to KITTI
    $ python convert.py --annotations_dir ~/datasets/handgun/annotations/pascal \
        --images_dir ~/datasets/handgun/images \
        --out_dir ~/datasets/handgun/kitti \
        --in_format pascal --out_format kitti \
        --kitti_ids_file file_ids.txt
        
    Usage: PASCAL to OpenImages
    $ python convert.py --annotations_dir ~/datasets/handgun/annotations/pascal \
        --images_dir ~/datasets/handgun/images \
        --out_dir ~/datasets/handgun/openimages \
        --in_format pascal --out_format openimages
    """

    # parse the command line arguments
    format_choices = ["coco", "darknet", "kitti", "openimages", "pascal"]
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--annotations_dir",
        required=True,
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
        "--out_dir",
        required=False,
        type=str,
        help="path to directory for output annotation files after conversion",
    )
    args_parser.add_argument(
        "--kitti_ids_file",
        required=False,
        type=str,
        help="name of the file that will contain the file IDs for KITTI, "
             "required if out_format is \"kitti\", will be written into the "
             "parent directory of the output directory for KITTI data",
    )
    args_parser.add_argument(
        "--in_format",
        required=True,
        type=str,
        choices=format_choices,
        help="format of input annotations",
    )
    args_parser.add_argument(
        "--out_format",
        required=True,
        type=str,
        choices=format_choices,
        help="format of output annotations",
    )
    args = vars(args_parser.parse_args())

    if args["in_format"] == "pascal":
        if args["out_format"] == "kitti":
            pascal_to_kitti(args["annotations_dir"], args["images_dir"], args["out_dir"], args["kitti_ids_file"])
        elif args["out_format"] == "openimages":
            pascal_to_openimages(args["annotations_dir"], args["images_dir"], args["out_dir"])
        elif args["out_format"] == "pascal":
            # if going from PASCAL to PASCAL then we're actually
            # converting the corresponding images from PNG to JPG
            pascal_png_to_jpg(args["annotations_dir"], args["images_dir"])
        else:
            raise ValueError(
                "Unsupported format conversion: "
                f"{args['in_format']} to {args['out_format']}",
            )
    elif args["in_format"] == "openimages":
        if args["out_format"] == "kitti":
            openimages_to_kitti(args["annotations_dir"], args["images_dir"], args["out_dir"], args["kitti_ids_file"])
        elif args["out_format"] == "pascal":
            pascal_to_openimages(args["annotations_dir"], args["images_dir"], args["out_dir"])
        elif args["out_format"] == "openimages":
            # if going from OpenImages to OpenImages then we're actually
            # converting the corresponding images from PNG to JPG
            openimages_png_to_jpg(args["annotations_dir"], args["images_dir"])
        else:
            raise ValueError(
                "Unsupported format conversion: "
                f"{args['in_format']} to {args['out_format']}",
            )
    else:
        raise ValueError(
            "Unsupported format conversion: "
            f"{args['in_format']} to {args['out_format']}",
        )
