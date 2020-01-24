import argparse
import collections
import concurrent.futures
import json
import logging
import math
import os
from typing import Dict

import cv2
import numpy as np
from PIL import Image
import six
import tensorflow as tf
from tensorflow.compat.v1.python_io import TFRecordWriter
from tqdm import tqdm

from cvdata.utils import image_dimensions, matching_ids


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
def _int64_list_feature(
        values,
) -> tf.train.Feature:
    """
    Returns a TF-Feature of int64_list.

    :param values:
    :return:
    """
    if not isinstance(values, collections.Iterable):
        values = [values]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


# ------------------------------------------------------------------------------
def _bytes_list_feature(
        values: str,
) -> tf.train.Feature:

    """
    Returns a TF-Feature of bytes.

    :param values a string
    :return TF-Feature of bytes
    """

    def norm2bytes(value):
        return value.encode() if isinstance(value, str) and six.PY3 else value

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


# ------------------------------------------------------------------------------
def _build_write_tfrecord(
    args: Dict,
):
    """
    Builds and writes a TFRecord with image and segmentation (mask) features.

    :param args: dictionary containing the following function arguments:
         output_path: the path of the TFRecord file to be written
         shard_id: shard ID (for multi-shard TFRecord datasets)
         num_per_shard: number of images/masks per shard
         num_images: total number of images in dataset
         filenames: file names for image files
         images_dir: directory containing image files
         masks_dir: directory containing mask files corresponding to the images
    """
    with TFRecordWriter(args["output_path"]) as tfrecord_writer:
        start_idx = args["shard_id"] * args["num_per_shard"]
        end_idx = min((args["shard_id"] + 1) * args["num_per_shard"], args["num_images"])
        for i in range(start_idx, end_idx):
            print(f'\r>> Converting image {i + 1}/{len(args["filenames"])} "'
                  f'shard {args["shard_id"]}')

            # Read the image.
            image = Image.open(os.path.join(args["images_dir"], args["filenames"][i]))
            img_bytes = image.tobytes()
            width, height = image.size

            # Read the semantic segmentation annotation.
            # TODO get the masks suffix from arguments
            masks_suffix = "_segmentation.png"
            mask_path = os.path.join(args["masks_dir"], os.path.splitext(args["filenames"][i])[0] + masks_suffix)
            mask = cv2.imread(mask_path)
            mask = cv2.split(mask)[0]
            mask_bytes = mask.tobytes()
            seg_height, seg_width = mask.shape
            if height != seg_height or width != seg_width:
                raise RuntimeError('Shape mismatched between image and label.')

            # Convert to tf example.
            example = tf.train.Example(features=tf.train.Features(feature={
                'image/encoded': _bytes_list_feature(img_bytes),
                'image/filename': _bytes_list_feature(args["filenames"][i]),
                'image/format': _bytes_list_feature('jpeg'),
                'image/height': _int64_list_feature(height),
                'image/width': _int64_list_feature(width),
                'image/channels': _int64_list_feature(3),
                'image/segmentation/class/encoded': (_bytes_list_feature(mask_bytes)),
                'image/segmentation/class/format': _bytes_list_feature('png'),
            }))
            tfrecord_writer.write(example.SerializeToString())


# ------------------------------------------------------------------------------
def masked_dataset_to_tfrecords(
        images_dir: str,
        masks_dir: str,
        tfrecord_dir: str,
        num_shards: int = 1,
        dataset_base_name: str = "tfrecord",
):
    masks_ext = ".png"
    images_ext = ".jpg"
    file_ids = matching_ids(masks_dir, images_dir, masks_ext, images_ext)
    num_images = len(file_ids)
    num_per_shard = int(math.ceil(num_images / num_shards))

    for shard_id in range(num_shards):
        output_filename = os.path.join(
            tfrecord_dir,
            f'{dataset_base_name}-{str(shard_id).zfill(5)}-of-{str(num_shards).zfill(5)}.tfrecord',
        )
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                print(f'\r>> Converting image {i + 1}/{len(file_ids)} shard {shard_id}')

                # read the image
                image_file_name = file_ids[i] + images_ext
                image_path = os.path.join(images_dir, image_file_name)
                image_data = tf.gfile.GFile(image_path, 'rb').read()
                width, height, _ = image_dimensions(image_path)

                # read the semantic segmentation annotation (mask)
                mask_path = os.path.join(masks_dir, file_ids[i] + masks_ext)
                seg_data = tf.gfile.GFile(mask_path, 'rb').read()
                seg_width, seg_height, _ = image_dimensions(mask_path)
                if height != seg_height or width != seg_width:
                    raise RuntimeError('Shape mismatched between image and mask.')

                # convert to a TensorFlow example
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image/encoded': _bytes_list_feature(image_data),
                    'image/filename': _bytes_list_feature(image_file_name),
                    'image/format': _bytes_list_feature('jpeg'),
                    'image/height': _int64_list_feature(height),
                    'image/width': _int64_list_feature(width),
                    'image/channels': _int64_list_feature(3),
                    'image/segmentation/class/encoded': (_bytes_list_feature(seg_data)),
                    'image/segmentation/class/format': _bytes_list_feature('png'),
                }))

                # write the example into the TFRecord (shard) file
                tfrecord_writer.write(example.SerializeToString())


# ------------------------------------------------------------------------------
def masks_to_tfrecords(
        images_dir: str,
        masks_dir: str,
        tfrecord_dir: str,
        num_shards: int = 1,
):
    """
    TODO

    :param images_dir: directory containing JPG image files
    :param masks_dir: directory containing PNG mask files
    :param tfrecord_dir: directory where TFRecord files will be written
    :param num_shards: number of TFRecord shards to create/write
    """

    # arguments validation
    if not os.path.exists(images_dir):
        raise ValueError(f"Invalid images directory path: {images_dir}")
    elif not os.path.exists(masks_dir):
        raise ValueError(f"Invalid masks directory path: {masks_dir}")

    # TODO make this an argument
    dataset = "tfrecord"

    # make the TFRecord(s) directory if it doesn't already exist
    os.makedirs(tfrecord_dir, exist_ok=True)

    filenames = []
    for file_name in os.listdir(images_dir):
        if file_name.endswith(".jpg"):
            filenames.append(file_name)

    num_images = len(filenames)
    num_per_shard = int(math.ceil(num_images / num_shards))

    args_iterable = []
    for shard_id in range(num_shards):
        output_filename = os.path.join(
            tfrecord_dir,
            f'{dataset}-{str(shard_id).zfill(5)}-of-{str(num_shards).zfill(5)}.tfrecord',
        )
        tfrecord_writing_args = {
            "output_path": output_filename,
            "shard_id": shard_id,
            "num_per_shard": num_per_shard,
            "num_images": num_images,
            "filenames": filenames,
            "images_dir": images_dir,
            "masks_dir": masks_dir,
        }
        args_iterable.append(tfrecord_writing_args)

    # use a ProcessPoolExecutor to download the images in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:

        # use the executor to map the download function to the iterable of arguments
        _logger.info(f"Building TFRecords in directory {tfrecord_dir} ")
        list(tqdm(executor.map(_build_write_tfrecord, args_iterable),
                  total=len(args_iterable)))


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

    :param images_dir: directory containing JPG image files
    :param annotations_file : annotation file containing segmentation (mask)
        regions, expected to be in the JSON format created by the VGG Annotator
        tool
    :param masks_dir: directory where PNG mask files will be written
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
        "--masks",
        required=False,
        type=str,
        help="path to directory where mask files will be written "
             "(or found if used as an input)",
    )
    args_parser.add_argument(
        "--tfrecords",
        required=False,
        type=str,
        help="path to directory where TFRecord output files will be written",
    )
    args_parser.add_argument(
        "--annotations",
        required=False,
        type=str,
        help="path to annotation file",
    )
    args_parser.add_argument(
        "--in_format",
        required=False,
        type=str,
        choices=["coco", "openimages", "png", "vgg"],
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
        required=False,
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
                args["masks"],
                args["classes"],
                args["combine"],
            )
    elif args["in_format"] == "png":
        if args["out_format"] == "tfrecord":
            masked_dataset_to_tfrecords(
                args["images"],
                args["masks"],
                args["tfrecords"],
                args["shards"],
            )
        else:
            raise ValueError(f"Unsupported output format: {args['out_format']}")

    else:
        raise ValueError(f"Unsupported input format: {args['in_format']}")


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
