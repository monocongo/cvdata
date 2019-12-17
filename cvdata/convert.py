import argparse
import concurrent.futures
import logging
import os
from pathlib import Path
import shutil
from typing import Dict, List, Set
from xml.etree import ElementTree

import cv2
from tqdm import tqdm

from cvdata.common import FORMAT_CHOICES
from cvdata.split import split_train_valid_test_images
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
    if kitti_ids_file_name is not None:
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
def images_png_to_jpg(
        images_dir: str,
):

    _logger.info(f"Converting all PNG files in directory {images_dir} to JPG")

    for file_name in tqdm(os.listdir(images_dir)):
        file_id, ext = os.path.splitext(file_name)
        if ext.lower() == ".png":
            png_file_path = os.path.join(images_dir, file_name)
            png_to_jpg(png_file_path, True)


# ------------------------------------------------------------------------------
def png_to_jpg(
        png_file_path: str,
        remove_png: bool = False,
) -> str:
    """
    Converts a PNG image to JPG and optionally removes the original PNG image file.

    :param png_file_path: path to a PNG image file
    :param remove_png: whether or not to remove the original PNG after the conversion
    :return: path to the new JPG file
    """

    # argument validation
    if not os.path.exists(png_file_path):
        raise ValueError(f"File does not exist: {png_file_path}")

    # read the PNG image data and rewrite as JPG
    jpg_file_path = os.path.splitext(png_file_path)[0] + ".jpg"
    img = cv2.imread(png_file_path)
    cv2.imwrite(jpg_file_path, img)
    if remove_png:
        os.remove(png_file_path)

    return jpg_file_path


# ------------------------------------------------------------------------------
def bounding_boxes_pascal(
        pascal_file_path: str,
) -> List[Dict]:
    """
    Get a list of bounding boxes from a PASCAL VOC annotation file.

    :param pascal_file_path:
    :return: list of bounding box dictionaries, with each dictionary containing
        five elements: "label" (string), "xmin", "ymin", "xmax", and "ymax" (ints)
    """

    boxes = []

    # load the contents of the annotations file into an ElementTree
    tree = ElementTree.parse(pascal_file_path)

    for obj in tree.iter("object"):

        label = obj.find("name").text
        bndbox = obj.find("bndbox")
        bbox_min_x = int(float(bndbox.find("xmin").text))
        bbox_min_y = int(float(bndbox.find("ymin").text))
        bbox_max_x = int(float(bndbox.find("xmax").text))
        bbox_max_y = int(float(bndbox.find("ymax").text))

        # make sure we don't have wonky values with mins > maxs
        if (bbox_min_x >= bbox_max_x) or (bbox_min_y >= bbox_max_y):

            # report the issue via log message
            _logger.warning("Bounding box minimum(s) greater than maximum(s)")

            # skip this box since it's not clear how to fix it
            continue

        # include this box in the list we'll return
        box = {
            "label": label,
            "xmin": bbox_min_x,
            "ymin": bbox_min_y,
            "xmax": bbox_max_x,
            "ymax": bbox_max_y,
        }
        boxes.append(box)

    return boxes


# ------------------------------------------------------------------------------
def pascal_to_openimages(
        pascal_dir: str,
        images_dir: str,
        openimages_dir: str,
        split: str = None,
):

    def csv_from_pascal(
            file_path_csv: str,
            images_directory: str,
            pascal_directory: str,
    ):
        bbox_file_ids = []
        with open(file_path_csv, "w") as csv_file:
            csv_file.write(
                "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,"
                "IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside,id,ClassName\n",
            )
            matching_file_ids = matching_ids(pascal_directory, images_directory, ".xml", ".jpg")
            for file_id in matching_file_ids:
                bboxes = bounding_boxes_pascal(
                    os.path.join(pascal_directory, file_id + ".xml"),
                )
                if len(bboxes):
                    bbox_file_ids.append(file_id)
                    for bbox in bboxes:
                        csv_file.write(
                            f"{file_id},,,,{bbox['xmin']},{bbox['xmax']},"
                            f"{bbox['ymin']},{bbox['ymax']},,,,,,,{bbox['label']}\n",
                        )

        # return list of image file IDs that are included in the CSV
        return set(bbox_file_ids)

    def remove_invalid_files(
            valid_file_ids: Set[str],
            directory: str,
    ):
        # go through the files in the images directory and remove any that
        # aren't included in the list of file IDs included in the CSV
        for name in os.listdir(directory):
            if os.path.splitext(name)[0] not in valid_file_ids:
                os.remove(os.path.join(directory, name))

    copy_image_files = True

    if split is not None:

        # create the destination directories for the image split subsets
        openimages_train_dir = os.path.join(openimages_dir, "train")
        openimages_valid_dir = os.path.join(openimages_dir, "validation")
        openimages_test_dir = os.path.join(openimages_dir, "test")
        os.makedirs(openimages_train_dir, exist_ok=True)
        os.makedirs(openimages_valid_dir, exist_ok=True)
        os.makedirs(openimages_test_dir, exist_ok=True)

        # split the images into the OpenImages split subdirectories
        train_percentage, valid_percentage, _ = map(float, split.split(":"))
        split_arguments = {
            "images_dir": images_dir,
            "train_images_dir": openimages_train_dir,
            "val_images_dir": openimages_valid_dir,
            "test_images_dir": openimages_test_dir,
            "train_percentage": train_percentage,
            "valid_percentage": valid_percentage,
            "copy_feature": copy_image_files,
        }
        split_train_valid_test_images(split_arguments)

        # translate PASCAL annotations to corresponding lines in the OpenImages CSVs
        for section in ["train", "validation", "test"]:

            csv_file_path = \
                os.path.join(openimages_dir, "sub-" + section + "-annotations-bbox.csv")
            if section == "train":
                split_images_dir = openimages_train_dir
            elif section == "valid":
                split_images_dir = openimages_valid_dir
            else:
                split_images_dir = openimages_test_dir
            file_ids = csv_from_pascal(csv_file_path, split_images_dir, pascal_dir)

            # go through the files in the images directory and remove any that
            # aren't included in the list of file IDs included in the CSV
            remove_invalid_files(file_ids, split_images_dir)

    else:

        # copy or move the image files into the OpenImages directory
        if copy_image_files:
            relocate = shutil.copy2
        else:
            relocate = shutil.move
        dest_images_dir = os.path.join(openimages_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        for file_name in os.listdir(images_dir):
            if file_name.endswith(".jpg"):
                image_file_path = os.path.join(images_dir, file_name)
                relocate(image_file_path, dest_images_dir)

        # write the annotations into the OpenImages CSV
        csv_file_path = \
            os.path.join(openimages_dir, "annotations-bbox.csv")
        file_ids = csv_from_pascal(csv_file_path, dest_images_dir, pascal_dir)

        # go through the files in the images directory and remove any that
        # aren't included in the list of file IDs included in the CSV
        remove_invalid_files(file_ids, dest_images_dir)


# ------------------------------------------------------------------------------
def openimages_to_kitti(
        annotations_csv: str,
        images_dir: str,
        out_dir: str,
        kitti_ids_file: str,
        split: str = None,
):
    # TODO
    pass


# ------------------------------------------------------------------------------
def openimages_to_pascal(
        annotations_csv: str,
        images_dir: str,
        out_dir: str,
        split: str = None,
):
    # TODO
    pass


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # Usage: PASCAL to KITTI
    # $ python convert.py --annotations_dir ~/datasets/handgun/annotations/pascal \
    #     --images_dir ~/datasets/handgun/images \
    #     --out_dir ~/datasets/handgun/kitti \
    #     --in_format pascal --out_format kitti \
    #     --kitti_ids_file file_ids.txt
    #
    # Usage: PASCAL to OpenImages
    # $ python convert.py --annotations_dir ~/datasets/handgun/pascal \
    #     --images_dir ~/datasets/handgun/images \
    #     --out_dir ~/datasets/handgun/openimages \
    #     --in_format pascal --out_format openimages
    #
    # Usage: bulk PNG to JPG image conversion
    # $ python convert.py --in_format png --out_format jpg \
    #     --images_dir /datasets/vehicle/usps

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
        "--split",
        required=False,
        type=str,
        help="split percentages, in format \"train:valid:test\" where train, "
             "valid and test are floats (decimals) and must sum to 1.0",
    )
    args_parser.add_argument(
        "--in_format",
        required=True,
        type=str,
        choices=FORMAT_CHOICES.append("png"),
        help="format of input annotations or images",
    )
    args_parser.add_argument(
        "--out_format",
        required=True,
        type=str,
        choices=FORMAT_CHOICES.append("jpg"),
        help="format of output annotations or images",
    )
    args = vars(args_parser.parse_args())

    if args["in_format"] == "pascal":
        if args["out_format"] == "kitti":
            pascal_to_kitti(
                args["annotations_dir"],
                args["images_dir"],
                args["out_dir"],
                args["kitti_ids_file"],
                args["split"],
            )
        elif args["out_format"] == "openimages":
            pascal_to_openimages(
                args["annotations_dir"],
                args["images_dir"],
                args["out_dir"],
                args["split"],
            )
        else:
            raise ValueError(
                "Unsupported format conversion: "
                f"{args['in_format']} to {args['out_format']}",
            )
    elif args["in_format"] == "openimages":
        if args["out_format"] == "kitti":
            openimages_to_kitti(
                args["annotations_dir"],
                args["images_dir"],
                args["out_dir"],
                args["kitti_ids_file"],
                args["split"],
            )
        elif args["out_format"] == "pascal":
            openimages_to_pascal(
                args["annotations_dir"],
                args["images_dir"],
                args["out_dir"],
                args["split"],
            )
        else:
            raise ValueError(
                "Unsupported format conversion: "
                f"{args['in_format']} to {args['out_format']}",
            )
    elif args["in_format"] == "png":
        if args["out_format"] == "jpg":
            images_png_to_jpg(args["images_dir"])
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
