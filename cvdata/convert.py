import argparse
from collections import namedtuple
import concurrent.futures
import logging
import os
from pathlib import Path
import shutil
from typing import Dict, List, NamedTuple, Set
from xml.etree import ElementTree

import contextlib2
import cv2
from object_detection.utils import dataset_util
from object_detection.dataset_tools import tf_record_creation_util
import pandas as pd
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

from cvdata.common import FORMAT_CHOICES
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
def _dataset_bbox_examples(
        images_dir: str,
        annotations_dir: str,
        annotation_format: str,
) -> pd.DataFrame:
    """

    :param images_dir: directory containing the dataset's *.jpg image files
    :param annotations_dir: directory containing the dataset's annotation files
    :param annotation_format: currently supported: "kitti" and "pascal"
    :return: pandas DataFrame with rows corresponding to the dataset's bounding boxes
    """

    # we expect all images to use the *.jpg extension
    image_ext = ".jpg"

    # list of bounding box annotations we'll eventually write to CSV
    bboxes = []

    if annotation_format == "pascal":

        # get the file IDs for all matching image/PASCAL pairs (i.e. the dataset)
        annotation_ext = ".xml"
        for file_id in matching_ids(
                annotations_dir,
                images_dir,
                annotation_ext,
                image_ext,
        ):
            # add all bounding boxes from the PASCAL file to the list of boxes
            pascal_path = os.path.join(annotations_dir, file_id + annotation_ext)
            tree = ElementTree.parse(pascal_path)
            root = tree.getroot()
            for member in root.findall('object'):
                bbox_values = (
                    root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text),
                )
                bboxes.append(bbox_values)

    elif annotation_format == "kitti":

        # get the file IDs for all matching image/PASCAL pairs (i.e. the dataset)
        annotation_ext = ".txt"
        for file_id in matching_ids(
                annotations_dir,
                images_dir,
                annotation_ext,
                image_ext,
        ):
            # get the image dimensions from the image file since this
            # info is not present in the corresponding KITTI annotation
            image_file_name = file_id + image_ext
            image_path = os.path.join(images_dir, image_file_name)
            width, height, _ = image_dimensions(image_path)

            # add all bounding boxes from the KITTI file to the list of boxes
            kitti_path = os.path.join(annotations_dir, file_id + annotation_ext)
            with open(kitti_path, "r") as kitti_file:
                for line in kitti_file:
                    kitti_box = line.split()
                    bbox_values = (
                        image_file_name,
                        width,
                        height,
                        kitti_box[0],
                        kitti_box[4],
                        kitti_box[5],
                        kitti_box[6],
                        kitti_box[7],
                    )
                    bboxes.append(bbox_values)

    else:
        raise ValueError(f"Unsupported annotation format: {annotation_format}")

    # stuff the bounding boxes into a pandas DataFrame
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    return pd.DataFrame(bboxes, columns=column_names)


# ------------------------------------------------------------------------------
def _create_tf_example(
        label_indices: Dict,
        group: NamedTuple,
        images_dir: str,
) -> tf.train.Example:
    """
    Creates a TensorFlow Example object representation of a group of annotations
    for an image file.

    :param label_indices: dictionary mapping class labels to their integer indices
    :param group: namedtuple containing filename and pd.Group values
    :param images_dir: directory containing dataset image files
    :return: TensorFlow Example object corresponding to the group of annotations
    """

    # read the image into a bytes object, get the dimensions
    image = Image.open(os.path.join(images_dir, group.filename))
    img_bytes = image.tobytes()
    width, height = image.size

    # lists of bounding box values for the example
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    # for each bounding box annotation add the values into the lists
    for index, row in group.object.iterrows():
        # normalize the bounding box coordinates to within the range (0, 1)
        xmins.append(int(row['xmin']) / width)
        xmaxs.append(int(row['xmax']) / width)
        ymins.append(int(row['ymin']) / height)
        ymaxs.append(int(row['ymax']) / height)
        # get the class label and corresponding index
        classes_text.append(row['class'].encode('utf8'))
        classes.append(label_indices[row['class']])

    # build the Example from the lists of coordinates, class labels/indices, etc.
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(img_bytes),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


# ------------------------------------------------------------------------------
def _generate_label_map(
        annotations_df: pd.DataFrame,
        labels_path: str,
) -> Dict:
    """

    :param annotations_df: pandas DataFrame with rows for annotations, should
        contain a column named "class" which contains the label text
    :param labels_path: path to label map prototxt file that will be written
    :return: dictionary of labels to indices represented by the label map
    """

    # make the directory where the file will be created, in case it doesn't yet exist
    os.makedirs(os.path.split(labels_path)[0], exist_ok=True)

    # dictionary of labels to indices that we'll populate and return
    label_indices = {}

    # create/write the file
    with open(labels_path, "w") as label_map_file:
        label_index = 1
        for label in annotations_df["class"].unique():
            item = "item {\n    id: " + f"{label_index}\n    name: '{label}'\n" + "}\n"
            label_map_file.write(item)
            label_index += 1

            label_indices[label] = label_index

    return label_indices


# ------------------------------------------------------------------------------
def _to_tfrecord(
        images_dir: str,
        annotations_dir: str,
        annotation_format: str,
        labels_path: str,
        tfrecord_path: str,
        total_shards: int,
):
    """
    Create TFRecord file(s) from an annotated dataset.

    :param images_dir: directory containing the dataset's image files
    :param annotations_dir: directory containing the dataset's annotation files
    :param annotation_format:
    :param labels_path: path to the label map prototext file that corresponds to
        the TFRecord files, and which will be generated by this function (will be
        overwritten if already exists)
    :param tfrecord_path: base TFRecord file path, files generated will have this
        as the base path with shard numbers at the end, for example if using 2 total
        shards then the resulting files will be <tfrecord_path>-00000-of-00002
        and <tfrecord_path>-00001-of-00002
    :param total_shards: number of shards over which to spread the records
    """

    # get the annotation "examples" as a pandas DataFrame
    examples_df = _dataset_bbox_examples(
        images_dir,
        annotations_dir,
        annotation_format,
    )

    # generate the prototext label map file
    label_indices = _generate_label_map(examples_df, labels_path)

    # group the annotation examples by corresponding file name
    data = namedtuple("data", ["filename", "object"])
    groupby = examples_df.groupby("filename")
    filename_groups = []
    for filename, x in zip(groupby.groups.keys(), groupby.groups):
        filename_groups.append(data(filename, groupby.get_group(x)))

    # write the TFRecords into the specified number of "shard" files
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = \
            tf_record_creation_util.open_sharded_output_tfrecords(
                tf_record_close_stack,
                tfrecord_path,
                total_shards,
            )
        for index, group in enumerate(filename_groups):
            tf_example = _create_tf_example(label_indices, group, images_dir)
            output_shard_index = index % total_shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


# ------------------------------------------------------------------------------
def kitti_to_tfrecord(
        images_dir: str,
        kitti_dir: str,
        labels_path: str,
        tfrecord_path: str,
        total_shards: int,
):
    """
    Create TFRecord file(s) from a KITTI-format annotated dataset.

    :param images_dir: directory containing the dataset's image files
    :param kitti_dir: directory containing the dataset's annotation files
    :param labels_path: path to the label map prototext file that corresponds to
        the TFRecord files, and which will be generated by this function (will be
        overwritten if already exists)
    :param tfrecord_path: base TFRecord file path, files generated will have this
        as the base path with shard numbers at the end, for example if using 2 total
        shards then the resulting files will be <tfrecord_path>-00000-of-00002
        and <tfrecord_path>-00001-of-00002
    :param total_shards: number of shards over which to spread the records
    """

    _logger.info("Converting images and annotations in KITTI format to TFRecord(s)")

    return _to_tfrecord(
        images_dir,
        kitti_dir,
        "kitti",
        labels_path,
        tfrecord_path,
        total_shards,
    )


# ------------------------------------------------------------------------------
def pascal_to_tfrecord(
        images_dir: str,
        pascal_dir: str,
        labels_path: str,
        tfrecord_path: str,
        total_shards: int,
):
    """
    Create TFRecord file(s) from a PASCAL-format annotated dataset.

    :param images_dir: directory containing the dataset's image files
    :param pascal_dir: directory containing the dataset's PASCAL annotation files
    :param labels_path: path to the label map prototext file that corresponds to
        the TFRecord files, and which will be generated by this function (will be
        overwritten if already exists)
    :param tfrecord_path: base TFRecord file path, files generated will have this
        as the base path with shard numbers at the end, for example if using 2 total
        shards then the resulting files will be <tfrecord_path>-00000-of-00002
        and <tfrecord_path>-00001-of-00002
    :param total_shards: number of shards over which to spread the records
    """

    _logger.info("Converting images and annotations in PASCAL format to TFRecord(s)")

    return _to_tfrecord(
        images_dir,
        pascal_dir,
        "pascal",
        labels_path,
        tfrecord_path,
        total_shards,
    )


# ------------------------------------------------------------------------------
def kitti_to_darknet(
        images_dir: str,
        kitti_dir: str,
        darknet_dir: str,
        darknet_labels: str,
):
    """
    Creates equivalent Darknet annotation files corresponding to a dataset with
    KITTI annotations.

    :param images_dir: directory containing the dataset's images
    :param kitti_dir: directory containing the dataset's KITTI annotation files
    :param darknet_dir: directory where the equivalent Darknet annotation files
        will be written
    :param darknet_labels: labels file corresponding to the label indices used
        in the Darknet annotation files, will be written into the specified
        Darknet annotations directory
    """

    _logger.info("Converting annotations in KITTI format to Darknet format equivalents")

    # create the Darknet annotations directory in case it doesn't yet exist
    os.makedirs(darknet_dir, exist_ok=True)

    # get list of file IDs of the KITTI annotations and corresponding images
    annotation_ext = ".txt"
    image_ext = ".jpg"
    file_ids = matching_ids(kitti_dir, images_dir, annotation_ext, image_ext)

    # dictionary of labels to indices
    label_indices = {}

    # build Darknet annotations from KITTI
    for file_id in tqdm(file_ids):

        # get the image's dimensions
        image_file_name = file_id + image_ext
        width, height, _ = image_dimensions(os.path.join(images_dir, image_file_name))

        # loop over all annotation lines in the KITTI file and compute Darknet equivalents
        annotation_file_name = file_id + annotation_ext
        with open(os.path.join(kitti_dir, annotation_file_name), "r") as kitti_file:
            darknet_bboxes = []
            for line in kitti_file:
                parts = line.split()
                label = parts[0]
                if label in label_indices:
                    label_index = label_indices[label]
                else:
                    label_index = len(label_indices)
                    label_indices[label] = label_index
                box_width_pixels = float(parts[6]) - float(parts[4]) + 1
                box_height_pixels = float(parts[7]) - float(parts[5]) + 1
                darknet_bbox = {
                    "label_index": label_index,
                    "center_x": ((box_width_pixels / 2) + float(parts[4])) / width,
                    "center_y": ((box_height_pixels / 2) + float(parts[5])) / height,
                    "box_width": box_width_pixels / width,
                    "box_height": box_height_pixels / height,
                }
                darknet_bboxes.append(darknet_bbox)

        # write the Darknet annotation boxes into a Darknet annotation file
        with open(os.path.join(darknet_dir, annotation_file_name), "w") as darknet_file:
            for darknet_bbox in darknet_bboxes:
                darknet_file.write(
                    f"{darknet_bbox['label_index']} {darknet_bbox['center_x']} "
                    f"{darknet_bbox['center_y']} {darknet_bbox['box_width']} "
                    f"{darknet_bbox['box_height']}\n"
                )

    # write the Darknet labels into a text file, one label per line,
    # in order according to the indices used in the annotation files
    with open(os.path.join(darknet_dir, darknet_labels), "w") as darknet_labels_file:
        index_labels = {v: k for k, v in label_indices.items()}
        for i in range(len(index_labels)):
            darknet_labels_file.write(f"{index_labels[i]}\n")


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
        kitti_ids_file_name: str = None,
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

    # write the KITTI IDs file in the KITTI directory's parent directory
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
        move_image_files: bool = False,
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

    # copy or move the image files into the OpenImages directory
    if move_image_files:
        relocate = shutil.move
    else:
        relocate = shutil.copy2
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
        move_images: bool = False,
):
    """
    TODO

    :param annotations_csv:
    :param images_dir:
    :param out_dir:
    :param kitti_ids_file:
    :param move_images:
    :return:
    """

    # TODO
    pass


# ------------------------------------------------------------------------------
def openimages_to_pascal(
        annotations_csv: str,
        images_dir: str,
        out_dir: str,
):
    """
    TODO

    :param annotations_csv:
    :param images_dir:
    :param out_dir:
    :return:
    """

    # TODO
    pass


# ------------------------------------------------------------------------------
def main():

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
    #
    # Usage: KITTI to TFRecord
    # $ python convert.py --in_format kitti --out_format tfrecord \
    #     --annotations_dir /data/kitti
    #     --images_dir /data/images
    #     --out_dir /data/dataset.record
    #     --tf_label_map /data/label_map.pbtxt

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
        "--tf_label_map",
        required=False,
        type=str,
        help="path to the protobuf text file containing the class label map "
             "used for TFRecords",
    )
    args_parser.add_argument(
        "--tf_shards",
        required=False,
        type=int,
        default=1,
        help="the number of shards to use when creating TFRecords",
    )
    args_parser.add_argument(
        "--move_kitti_images",
        default=False,
        action='store_true',
        help="move image files into the KITTI images directory, rather than copying",
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
    args_parser.add_argument(
        "--darknet_labels",
        required=False,
        type=str,
        help="file name of the labels file that will correspond to the label "
             "indices used in the Darknet annotation files, to be written "
             "in the Darknet directory",
    )
    args = vars(args_parser.parse_args())

    if args["in_format"] == "pascal":
        if args["out_format"] == "kitti":
            pascal_to_kitti(
                args["annotations_dir"],
                args["images_dir"],
                args["out_dir"],
                args["kitti_ids_file"],
                args["move_kitti_images"],
            )
        elif args["out_format"] == "openimages":
            pascal_to_openimages(
                args["annotations_dir"],
                args["images_dir"],
                args["out_dir"],
                args["move_kitti_images"],
            )
        elif args["out_format"] == "tfrecord":
            pascal_to_tfrecord(
                args["images_dir"],
                args["annotations_dir"],
                args["tf_label_map"],
                args["out_dir"],
                args["tf_shards"],
            )
        else:
            raise ValueError(
                "Unsupported format conversion: "
                f"{args['in_format']} to {args['out_format']}",
            )
    elif args["in_format"] == "kitti":
        if args["out_format"] == "darknet":
            kitti_to_darknet(
                args["images_dir"],
                args["annotations_dir"],
                args["out_dir"],
                args["darknet_labels"],
            )
        elif args["out_format"] == "tfrecord":
            kitti_to_tfrecord(
                args["images_dir"],
                args["annotations_dir"],
                args["tf_label_map"],
                args["out_dir"],
                args["tf_shards"],
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
                args["move_kitti_images"],
            )
        elif args["out_format"] == "pascal":
            openimages_to_pascal(
                args["annotations_dir"],
                args["images_dir"],
                args["out_dir"],
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


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    main()
