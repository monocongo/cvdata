import argparse
import concurrent.futures
import fileinput
import logging
import os
from typing import Dict
from xml.etree import ElementTree

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
def rename_label_kitti(arguments: Dict):

    with fileinput.FileInput(arguments["file_path"], inplace=True) as file_input:
        for line in file_input:
            line = line.rstrip("\r\n")
            print(line.replace(arguments["old"], arguments["new"]))


# ------------------------------------------------------------------------------
def rename_label_pascal(arguments: Dict):

    # load the contents of the annotations file into an ElementTree
    pascalxml_path = arguments["file_path"]
    tree = ElementTree.parse(pascalxml_path)

    # remove extraneous newlines and whitespace from text elements
    for element in tree.getiterator():
        if element.text:
            element.text = element.text.strip()

    # loop over all objects, renaming those that are relevant
    for obj in tree.iter("object"):

        name = obj.find("name")
        if name and (name.text.strip() == arguments["old"]):
            name.text = arguments["new"]

    # write the annotation document back into the annotation file
    tree.write(pascalxml_path)


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--labels_dir",
        required=True,
        type=str,
        help="directory containing images to be renamed",
    )
    args_parser.add_argument(
        "--old",
        required=True,
        type=str,
        help="original label name that should be updated",
    )
    args_parser.add_argument(
        "--new",
        required=True,
        type=str,
        help="new label name that should replace the original",
    )
    args_parser.add_argument(
        "--format",
        required=True,
        type=str,
        default="kitti",
        choices=["coco", "darknet", "kitti", "pascal"],
        help="format of annotations to be renamed",
    )
    args = vars(args_parser.parse_args())

    if args["format"] == "kitti":
        file_ext = ".txt"
        rename_function = rename_label_kitti
    elif args["format"] == "pascal":
        file_ext = ".txt"
        rename_function = rename_label_kitti
    else:
        raise ValueError("Only KITTI and PASCAL annotation files are supported")

    # create an iterable of renaming function arguments
    # that we'll later map to the appropriate rename function
    rename_arguments_list = []
    file_names = [each for each in os.listdir(args["labels_dir"]) if each.endswith(file_ext)]
    for file_name in file_names:

        rename_arguments = {
            "old": args["old"],
            "new": args["new"],
            "file_path": os.path.join(args["labels_dir"], file_name),
        }
        rename_arguments_list.append(rename_arguments)

    # use a ProcessPoolExecutor to rename the labels in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:

        # use the executor to map the download function to the iterable of arguments
        _logger.info(
            f"Renaming all KITTI labels in directory {args['labels_dir']} "
            f"from {args['old']} to {args['new']}",
        )
        list(tqdm(executor.map(rename_function, rename_arguments_list),
                  total=len(rename_arguments_list)))
