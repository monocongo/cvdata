import argparse
import concurrent.futures
import fileinput
import logging
import os
from typing import Dict
from xml.etree import ElementTree

from tqdm import tqdm

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
def relabel_kitti(
        file_path: str,
        old_label: str,
        new_label: str,
):
    """
    Replaces the label values of a KITTI annotation file.

    :param file_path: path of the KITTI file to have labels replaced
    :param old_label: label value which if found will be replaced by the new label
    :param new_label: new label value
    """

    # arguments validation
    _validate_args(file_path, old_label, new_label)

    # replace the label in-place
    with fileinput.FileInput(file_path, inplace=True) as file_input:
        for line in file_input:
            line = line.rstrip("\r\n")
            print(line.replace(old_label, new_label))


# ------------------------------------------------------------------------------
def relabel_pascal(
        file_path: str,
        old_label: str,
        new_label: str,
):
    """
    Replaces the label values of a PASCAL VOC annotation file.

    :param file_path: path of the PASCAL VOC file to have labels replaced
    :param old_label: label value which if found will be replaced by the new label
    :param new_label: new label value
    """

    # arguments validation
    _validate_args(file_path, old_label, new_label)

    # load the contents of the annotations file into an ElementTree
    tree = ElementTree.parse(file_path)

    # loop over all objects, relabeling those that are relevant
    for obj in tree.iter("object"):

        name = obj.find("name")
        if (name is not None) and (name.text.strip() == old_label):
            name.text = new_label

    # write the annotation document back into the annotation file
    tree.write(file_path)


# ------------------------------------------------------------------------------
def _validate_args(
        file_path: str,
        old_label: str,
        new_label: str,
):

    if file_path is None:
        raise ValueError("Missing the file path argument")
    elif old_label is None:
        raise ValueError("Missing the old label argument")
    elif new_label is None:
        raise ValueError("Missing the new label argument")
    elif not os.path.isfile(file_path):
        raise ValueError(f"File path argument {file_path} is not a valid file path")


# ------------------------------------------------------------------------------
def _relabel_kitti(arguments: Dict):
    """
    Unpacks a dictionary of arguments and calls the function for replacing the
    labels of a KITTI annotation file.

    :param arguments: dictionary of function arguments, should include:
         "file_path": path of the KITTI file to have labels renamed
         "old": label name which if found will be renamed
         "new": new label name value
    """

    relabel_kitti(arguments["file_path"], arguments["old"], arguments["new"])


# ------------------------------------------------------------------------------
def _relabel_pascal(arguments: Dict):
    """
    Unpacks a dictionary of arguments and calls the function for renaming the
    labels of a PASCAL VOC annotation file.

    :param arguments: dictionary of function arguments, should include:
        "file_path": path of the KITTI file to have labels renamed
        "old": label name which if found will be renamed
        "new": new label name value
    """

    relabel_pascal(arguments["file_path"], arguments["old"], arguments["new"])


# ------------------------------------------------------------------------------
def main():

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
        choices=FORMAT_CHOICES,
        help="annotation format of the annotations to be relabeled",
    )
    args = vars(args_parser.parse_args())

    if args["format"] == "kitti":
        file_ext = ".txt"
        relabel_function = _relabel_kitti
    elif args["format"] == "pascal":
        file_ext = ".xml"
        relabel_function = _relabel_pascal
    else:
        raise ValueError("Only KITTI and PASCAL annotation files are supported")

    # create an iterable of renaming function arguments
    # that we'll later map to the appropriate relabel function
    relabel_arguments_list = []
    file_names = [each for each in os.listdir(args["labels_dir"]) if each.endswith(file_ext)]
    for file_name in file_names:

        relabel_arguments = {
            "old": args["old"],
            "new": args["new"],
            "file_path": os.path.join(args["labels_dir"], file_name),
        }
        relabel_arguments_list.append(relabel_arguments)

    # use a ProcessPoolExecutor to replace the labels in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:

        # use the executor to map the relabel function to the iterable of arguments
        _logger.info(
            f"Replacing all annotation labels in directory {args['labels_dir']} "
            f"from {args['old']} to {args['new']}",
        )
        list(tqdm(executor.map(relabel_function, relabel_arguments_list),
                  total=len(relabel_arguments_list)))


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # Usage:
    # $ python relabel.py --labels_dir /data/cvdata/pascal \
    #   --old handgun --new firearm --format pascal

    main()
