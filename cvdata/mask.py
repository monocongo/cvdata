import argparse
import json
import logging
import os
from typing import Dict

import cv2
import numpy as np
from tqdm import tqdm

from cvdata.utils import image_dimensions


# ------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def _class_labels_to_ids(
        labels_path: str,
) -> Dict:
    """
    Reads a text file, which is assumed to contain one class label per line, and
    returns a dictionary with class labels as keys mapped to the class ID (i.e.
    the label's line number).

    So a labels file like so:

    cat
    dog
    panda

    will result in a dictionary like so:

    {
      "cat": 1,
      "dog": 2,
      "panda": 3,
    }

    :param labels_path: path to a file containing class labels used in
        a segmentation dataset, with one class label per line
    :return: dictionary mapping class labels to ID values
    """

    class_labels = {}
    with open(labels_path, "r") as class_labels_file:
        class_id = 1
        for class_label in class_labels_file:
            class_labels[class_label.strip()] = class_id
            class_id += 1

    return class_labels


# ------------------------------------------------------------------------------
def vgg_to_tfrecords(
        images_dir: str,
        annotations_file: str,
        tfrecord_dir: str,
        class_labels_file: str,
        num_shards: int = 1,
):
    """
    TODO

    :param images_dir: directory containing image files to be filtered
    :param annotations_file : annotation file containing segmentation (mask)
        regions, expected to be in the JSON format created by the VGG Annotator
        tool
    :param tfrecord_dir: directory where TFRecord files will be written
    :param class_labels_file: text file containing one class label per line
    :param num_shards: number of TFRecord shards to create/write
    """

    # arguments validation
    if not os.path.exists(images_dir):
        raise ValueError(f"Invalid images directory path: {images_dir}")
    elif not os.path.exists(annotations_file):
        raise ValueError(f"Invalid annotations file path: {annotations_file}")

    # make the TFRecord(s) directory if it doesn't already exist
    os.makedirs(tfrecord_dir, exist_ok=True)

    # load the contents of the annotation JSON file (created
    # using the VIA tool) and initialize the annotations dictionary
    annotations = json.loads(open(annotations_file).read())
    image_annotations = {}

    # loop over the file ID and annotations themselves (values)
    for data in annotations.values():

        # store the data in the dictionary using the filename as the key
        image_annotations[data["filename"]] = data

    # get a dictionary of class labels to class IDs
    class_labels = _class_labels_to_ids(class_labels_file)


# ------------------------------------------------------------------------------
def vgg_to_masks(
        images_dir: str,
        annotations_file: str,
        masks_dir: str,
        class_labels_file: str,
        combine_into_one: bool = False,
):
    """
    TODO

    :param images_dir: directory containing image files to be filtered
    :param annotations_file : annotation file containing segmentation (mask)
        regions, expected to be in the JSON format created by the VGG Annotator
        tool
    :param masks_dir: directory where mask files will be written
    :param class_labels_file: text file containing one class label per line
    :param combine_into_one: if True then combine all mask regions for an image
        into a single mask file
    """

    # arguments validation
    if not os.path.exists(images_dir):
        raise ValueError(f"Invalid images directory path: {images_dir}")
    elif not os.path.exists(annotations_file):
        raise ValueError(f"Invalid annotations file path: {annotations_file}")

    # make the masks directory if it doesn't already exist
    os.makedirs(masks_dir, exist_ok=True)

    # load the contents of the annotation JSON file (created
    # using the VIA tool) and initialize the annotations dictionary
    annotations = json.loads(open(annotations_file).read())
    image_annotations = {}

    # loop over the file ID and annotations themselves (values)
    for data in annotations.values():

        # store the data in the dictionary using the filename as the key
        image_annotations[data["filename"]] = data

    # get a dictionary of class labels to class IDs
    class_labels = _class_labels_to_ids(class_labels_file)

    _logger.info("Generating mask files...")
    for image_file_name in tqdm(os.listdir(images_dir)):

        # skip any files without a *.jpg extension
        if not image_file_name.endswith(".jpg"):
            continue

        file_id = os.path.splitext(image_file_name)[0]

        # grab the image info and then grab the annotation data for
        # the current image based on the unique image ID
        annotation = image_annotations[image_file_name]

        # get the image's dimensions
        width, height, _ = image_dimensions(os.path.join(images_dir, image_file_name))

        # if combining all regions into a single mask file
        # then we'll only need to allocate the mask array once
        if combine_into_one:
            # allocate memory for the region mask
            region_mask = np.zeros((height, width, 3), dtype="uint8")

        # loop over each of the annotated regions
        for (i, region) in enumerate(annotation["regions"]):

            # if not combining all regions into a single mask file then
            # we'll need to reallocate the mask array for each mask region
            if not combine_into_one:
                # allocate memory for the region mask
                region_mask = np.zeros((height, width, 3), dtype="uint8")

            # grab the shape and region attributes
            shape_attributes = region["shape_attributes"]
            region_attributes = region["region_attributes"]

            # find the class ID corresponding to the region's class attribute
            class_label = region_attributes["class"]
            if class_label not in class_labels:
                raise ValueError(
                    "No corresponding class ID found for the class label "
                    f"found in the region attributes -- label: {class_label}",
                )
            else:
                class_id = class_labels[class_label]

            # get the array of (x, y)-coordinates for the region's mask polygon
            x_coords = shape_attributes["all_points_x"]
            y_coords = shape_attributes["all_points_y"]
            coords = zip(x_coords, y_coords)
            poly_coords = [[x, y] for x, y in coords]
            pts = np.array(poly_coords, np.int32)

            # reshape the points to (<# of coordinates>, 1, 2)
            pts = pts.reshape((-1, 1, 2))

            # draw the polygon mask, using the class ID as the mask value
            cv2.fillPoly(region_mask, [pts], color=[class_id]*3)

            # if not combining all masks into a single file
            # then write this mask into its own file
            if not combine_into_one:
                # write the mask file
                mask_file_name = f"{file_id}_segmentation_{i}.png"
                cv2.imwrite(os.path.join(masks_dir, mask_file_name), region_mask)

        # write a combined mask file, if requested
        if combine_into_one:
            # write the mask file
            mask_file_name = f"{file_id}_segmentation.png"
            cv2.imwrite(os.path.join(masks_dir, mask_file_name), region_mask)

    _logger.info("Done")


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
        "--out_dir",
        required=True,
        type=str,
        help="path to directory where output files should be written",
    )
    args_parser.add_argument(
        "--annotations",
        required=True,
        type=str,
        help="path to annotation file",
    )
    args_parser.add_argument(
        "--in_format",
        required=False,
        type=str,
        choices=["vgg", "coco", "openimages"],
        help="format of input annotations",
    )
    args_parser.add_argument(
        "--out_format",
        required=False,
        type=str,
        choices=["png", "tfrecord"],
        help="format of output annotations/masks",
    )
    args_parser.add_argument(
        "--classes",
        required=True,
        type=str,
        help="path of the class labels file listing one class per line",
    )
    args_parser.add_argument(
        "--combine",
        default=False,
        action='store_true',
        help="combine all regions/classes into a single mask file",
    )
    args_parser.add_argument(
        "--shards",
        required=False,
        default=1,
        type=int,
        help="number of shard files to use when converting to TFRecord format",
    )
    args = vars(args_parser.parse_args())

    if args["in_format"] == "vgg":
        if args["out_format"] == "png":
            vgg_to_masks(
                args["images"],
                args["annotations"],
                args["out_dir"],
                args["classes"],
                args["combine"],
            )
        elif args["out_format"] == "tfrecord":
            vgg_to_tfrecord(
                args["images"],
                args["annotations"],
                args["out_dir"],
                args["classes"],
                args["shards"],
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
