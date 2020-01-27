import argparse
import logging
import os
import shutil
from typing import Dict, Set

from tqdm import tqdm

from cvdata.common import FORMAT_CHOICES, FORMAT_EXTENSIONS
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
def _darknet_indices_to_labels(
        darknet_labels_path: str,
) -> Dict:
    """
    TODO

    :param darknet_labels_path: path to the file containing labels used in
        the Darknet dataset, should correspond to the labels used in the Darknet
        annotation file
    :return: dictionary mapping index values to corresponding labels text
    """

    index_labels = {}
    with open(darknet_labels_path, "r") as darknet_labels_file:
        index = 0
        for line in darknet_labels_file:
            darknet_label = line.split()[0]
            index_labels[index] = darknet_label
            index += 1

    return index_labels


# ------------------------------------------------------------------------------
def _count_boxes_darknet(
        darknet_file_path: str,
        darknet_label_indices: Dict,
) -> Dict:
    """
    Counts the boxes in a KITTI annotation file and returns a dictionary mapping
    the labels to the corresponding count of bounding boxes with that label.

    :param darknet_file_path: path to a Darknet annotation file
    :param darknet_label_indices: dictionary of label indices to corresponding
        label texts
    :return: dictionary with labels mapped to their bounding box counts
    """

    box_counts = {}
    with open(darknet_file_path) as darknet_file:
        for bbox_line in darknet_file:
            parts = bbox_line.split()
            label_index = int(parts[0])
            if label_index not in darknet_label_indices:
                raise ValueError(
                    f"Unexpected label index ({label_index} found in Darknet "
                    f"annotation file {darknet_file_path}, incompatible with "
                    f"the provided Darknet labels file")
            else:
                darknet_label = darknet_label_indices[label_index]
                if darknet_label in box_counts:
                    box_counts[darknet_label] = box_counts[darknet_label] + 1
                else:
                    box_counts[darknet_label] = 1

    return box_counts


# ------------------------------------------------------------------------------
def _count_boxes_kitti(
        kitti_file_path: str,
) -> Dict:
    """
    Counts the boxes in a KITTI annotation file and returns a dictionary mapping
    the labels to the corresponding count of bounding boxes with that label.

    :param kitti_file_path: path to a KITTI annotation file
    :return: dictionary with labels mapped to their bounding box counts
    """

    box_counts = {}
    with open(kitti_file_path) as kitti_file:
        for bbox_line in kitti_file:
            parts = bbox_line.split()
            kitti_label = parts[0]
            if kitti_label in box_counts:
                box_counts[kitti_label] = box_counts[kitti_label] + 1
            else:
                box_counts[kitti_label] = 1

    return box_counts


# ------------------------------------------------------------------------------
def _count_boxes(
        annotation_file_path: str,
        annotation_format: str,
        darknet_label_indices: Dict = None,
) -> Dict:
    """
    Counts the number of bounding boxes in an annotation file, per label.

    :param annotation_file_path: the annotation file from which we'll count
        the number of bounding boxes per label
    :param annotation_format: format of the annotation files
    :param darknet_label_indices: dictionary of labels to the indices used
        within Darknet files, only relevant/required for Darknet format
    :return: dictionary mapping labels to the number of bounding boxes
        in the annotation file for that class label
    """

    if annotation_format == "darknet":
        return _count_boxes_darknet(annotation_file_path, darknet_label_indices)
    elif annotation_format == "kitti":
        return _count_boxes_kitti(annotation_file_path)
    else:
        raise ValueError(f"Unsupported annotation format: {annotation_format}")


# ------------------------------------------------------------------------------
def _write_with_removed_labels_darknet(
        src_darknet_path,
        dest_darknet_path,
        darknet_valid_indices: Set,
):
    """
    Writes a Darknet annotation file containing only valid label indices,
    essentially a copy of the source Darknet file without any bounding boxes
    that contain "non-valid" indices.

    :param src_darknet_path:
    :param dest_darknet_path:
    :param darknet_valid_indices:
    """

    with open(dest_darknet_path, "w") as dest_darknet_file:
        with open(src_darknet_path, "r") as src_darknet_file:
            for line in src_darknet_file:
                label_index = int(line.split()[0])
                if label_index in darknet_valid_indices:
                    dest_darknet_file.write(line)


# ------------------------------------------------------------------------------
def _write_with_removed_labels_kitti(
        src_kitti_path,
        dest_kitti_path,
        valid_labels,
):
    """
    Writes a KITTI annotation file containing only valid label labels,
    essentially a copy of the source KITTI file without any bounding boxes
    that contain "non-valid" labels.

    :param src_kitti_path:
    :param dest_kitti_path:
    :param valid_labels:
    """
    with open(dest_kitti_path, "w") as dest_kitti_file:
        with open(src_kitti_path, "r") as src_kitti_file:
            for line in src_kitti_file:
                kitti_label = line.split()[0]
                if kitti_label in valid_labels:
                    dest_kitti_file.write(line)


# ------------------------------------------------------------------------------
def _write_with_removed_labels(
        src_annotation_path,
        dest_annotation_path,
        annotation_format,
        valid_labels: Set = None,
        darknet_valid_indices: Set = None,
):
    """
    Writes an annotation file containing only valid label labels,
    essentially a copy of the source annotation file without any bounding boxes
    that contain "non-valid" labels or indices.

    :param src_annotation_path:
    :param dest_annotation_path:
    :param annotation_format:
    :param valid_labels:
    :param darknet_valid_indices:
    """

    if annotation_format == "darknet":
        _write_with_removed_labels_darknet(
            src_annotation_path,
            dest_annotation_path,
            darknet_valid_indices,
        )
    elif annotation_format == "kitti":
        _write_with_removed_labels_kitti(
            src_annotation_path,
            dest_annotation_path,
            valid_labels,
        )
    else:
        raise ValueError(f"Unsupported annotation format: {annotation_format}")


# ------------------------------------------------------------------------------
def filter_class_boxes(
        src_images_dir: str,
        src_annotations_dir: str,
        dest_images_dir: str,
        dest_annotations_dir: str,
        class_label_counts: Dict,
        annotation_format: str,
        darknet_labels_path: str = None,
):
    """
    TODO

    :param src_images_dir:
    :param src_annotations_dir:
    :param dest_images_dir:
    :param dest_annotations_dir:
    :param class_label_counts:
    :param annotation_format:
    :param darknet_labels_path: path to the labels file corresponding
        to a Darknet-annotated dataset
    """

    _logger.info(
        f"Filtering dataset into annotations directory {dest_annotations_dir} "
        f"and images directory {dest_images_dir}"
    )

    # make sure we don't have the same directories for src and dest
    if src_images_dir == dest_images_dir:
        raise ValueError(
            "Source and destination image directories are "
            "the same, must be different",
        )
    if src_annotations_dir == dest_annotations_dir:
        raise ValueError(
            "Source and destination annotation directories are "
            "the same, must be different",
        )

    # determine the file extension to be used for annotations
    if annotation_format not in ["darknet", "kitti"]:
        raise ValueError(f"Unsupported annotation format: {annotation_format}")
    else:
        annotation_ext = FORMAT_EXTENSIONS[annotation_format]
    image_ext = ".jpg"

    # make the destination directories, in case they don't already exist
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_annotations_dir, exist_ok=True)

    # keep a count of the boxes we've processed for each image class type
    processed_class_counts = {k: 0 for k in class_label_counts.keys()}

    # get all file IDs for image/annotation file matches
    file_ids = \
        matching_ids(
            src_annotations_dir,
            src_images_dir,
            annotation_ext,
            image_ext,
        )

    # only include the labels specified in the class counts dictionary
    valid_labels = set(class_label_counts.keys())

    # if we're processing Darknet annotations then read the labels file to get
    # a mapping of indices to labels used with the Darknet annotation files
    darknet_valid_indices = None
    darknet_index_labels = None
    if annotation_format == "darknet":
        # read the Darknet labels into a dictionary mapping label to label index
        darknet_index_labels = _darknet_indices_to_labels(darknet_labels_path)

        # get the set of valid indices, i.e. all Darknet indices
        # corresponding to the labels to be included in the filtered dataset
        darknet_valid_indices = set()
        for index, darknet_label in darknet_index_labels.items():
            if darknet_label in valid_labels:
                darknet_valid_indices.add(index)

    # loop over all the possible image/annotation file pairs
    for file_id in tqdm(file_ids):

        # only include the file if we find a box for one of the specified labels
        include_file = False

        # if any labels are found in the annotation file that aren't included
        # in the list of image classes to filter then we'll want to remove the
        # boxes from the annotation file before writing to the destination directory
        irrelevant_labels_found = False

        # get the count(s) of boxes per class label
        annotation_file_name = file_id + annotation_ext
        src_annotation_path = os.path.join(src_annotations_dir, annotation_file_name)
        box_counts = _count_boxes(src_annotation_path, annotation_format, darknet_index_labels)

        for class_label in box_counts.keys():
            if class_label in class_label_counts:
                processed_class_count = processed_class_counts[class_label]
                if processed_class_counts[class_label] < class_label_counts[class_label]:
                    include_file = True
                    processed_class_counts[class_label] = processed_class_count + box_counts[class_label]
            else:
                irrelevant_labels_found = True

        if include_file:
            dest_annotation_path = os.path.join(dest_annotations_dir, annotation_file_name)
            if irrelevant_labels_found:
                # remove the unnecessary labels or indices from the
                # annotation and write it into the destination directory
                _write_with_removed_labels(
                    src_annotation_path,
                    dest_annotation_path,
                    annotation_format,
                    valid_labels,
                    darknet_valid_indices,
                )
            else:
                # copy te annotation file into the destination directory as-is
                shutil.copy(src_annotation_path, dest_annotation_path)

            # copy the source image file into the destination images directory
            image_file_name = file_id + image_ext
            src_image_path = os.path.join(src_images_dir, image_file_name)
            dest_image_path = os.path.join(dest_images_dir, image_file_name)
            shutil.copy(src_image_path, dest_image_path)


# ------------------------------------------------------------------------------
def main():

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--src_annotations",
        required=True,
        type=str,
        help="path to directory containing original dataset's annotation files",
    )
    args_parser.add_argument(
        "--src_images",
        required=True,
        type=str,
        help="path to directory containing original dataset's image files",
    )
    args_parser.add_argument(
        "--dest_annotations",
        required=True,
        type=str,
        help="path to directory where the filtered dataset's annotation "
             "files will be written",
    )
    args_parser.add_argument(
        "--dest_images",
        required=True,
        type=str,
        help="path to directory where the filtered dataset's image files "
             "will be written",
    )
    args_parser.add_argument(
        "--format",
        required=True,
        type=str,
        choices=FORMAT_CHOICES,
        help="format of the annotations",
    )
    args_parser.add_argument(
        "--darknet_labels",
        required=False,
        type=str,
        help="path to the Darknet labels file, only relevant if processing "
             "a dataset with Darknet annotations (i.e. --format darknet)",
    )
    args_parser.add_argument(
        "--boxes_per_class",
        required=True,
        type=str,
        nargs="*",
        help="counts of boxes per image class type, in format class:count "
             "(multiple counts are space separated)",
    )
    args = vars(args_parser.parse_args())

    # make a dictionary of class labels mapped to their maximum box counts
    class_counts = {}
    for class_count in args["boxes_per_class"]:
        label, count = class_count.split(":")
        class_counts[label] = int(count)

    # filter the dataset by class/count
    filter_class_boxes(
        args["src_images"],
        args["src_annotations"],
        args["dest_images"],
        args["dest_annotations"],
        class_counts,
        args["format"],
        args["darknet_labels"],
    )


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # Usage: filter a dataset down to a collection fo image and annotation files
    #        so that it contains a specified number of annotations (bounding
    #        boxes) per image class type
    #
    # $ python filter.py --format kitti --src_images /data/original/images \
    # >   --src_annotations /data/original/kitti \
    # >   --dest_images /data/filtered/images \
    # >   --dest_annotations /data/filtered/kitti \
    # >   --boxes_per_class dog:5000 cat:5000 panda:5000

    main()
