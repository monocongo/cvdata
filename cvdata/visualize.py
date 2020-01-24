import argparse
import json
import logging
import os
from typing import Dict, List
from xml.etree import ElementTree

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from cvdata.common import FORMAT_CHOICES
from cvdata.resize import resize_with_padding


# enable TensorFlow eager execution (only required for versions before 2.0)
if int(tf.__version__[0]) < 2:
    tf.enable_eager_execution()

# colors for annotation boxes/labels
_RECTANGLE_BGR = (0, 255, 0)
_TEXT_BGR = (255, 0, 255)

# ------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def show_tfrecords_tlt(
        tfrecords_dir: str,
        image_directory: str,
):
    """
    Displays images with bounding box annotations from all TFRecord files in a
    directory containing TFRecord files and an associated images directory
    containing the corresponding image files.

    The TFRecord format is assumed to be the one used by the NVIDIA Transfer
    Learning Toolkit's KITTI to TFRecord dataset conversion tool, described here:
    https://docs.nvidia.com/metropolis/TLT/tlt-getting-started-guide/index.html#conv_tfrecords_topic

    :param tfrecords_dir: directory containing TFRecord files
    :param image_directory: directory containing image files corresponding to
        the examples contained within the vairious TFRecord files
    """

    image_count = 0
    for tfrecords_file_name in os.listdir(tfrecords_dir):

        # parse each TFRecord file
        tfrecords_file_path = os.path.join(tfrecords_dir, tfrecords_file_name)
        tf_dataset = tf.data.TFRecordDataset(tfrecords_file_path)
        for record in tf_dataset:

            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            feature = example.features.feature
            frame_id = os.path.split(str(feature['frame/id'].bytes_list.value[0])[2:-1])[-1]
            object_class_id = feature['target/object_class'].bytes_list.value
            x_min = feature['target/coordinates_x1'].float_list.value
            x_max = feature['target/coordinates_x2'].float_list.value
            y_min = feature['target/coordinates_y1'].float_list.value
            y_max = feature['target/coordinates_y2'].float_list.value
            i = 0
            current_image_path = str(os.path.join(image_directory, frame_id)) + '.jpg'
            img = cv2.imread(current_image_path)
            while i < len(x_min):
                # draw bounding box
                cv2.rectangle(img, (int(x_min[i]), int(y_min[i])),
                              (int(x_max[i]), int(y_max[i])),
                              _RECTANGLE_BGR, 2)
                # draw the label
                cv2.putText(
                    img,
                    object_class_id[i].decode(),
                    (int(x_min[i] + 3), int(y_min[i]) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    _TEXT_BGR,
                    1,
                )
                i += 1

            # show the output image
            _logger.info(f"{image_count} Displaying {len(x_min)} boxes for {current_image_path}")
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            image_count += 1

    cv2.destroyAllWindows()


# ------------------------------------------------------------------------------
def show_tfrecords_tfod(
        tfrecords_dir: str,
):
    """
    Displays images with bounding box annotations from all TFRecord files in a
    directory containing TFRecord files and an associated images directory
    containing the corresponding image files.

    The TFRecord format is assumed to be the one used as input for TensorFlow
    object detection models, described here:
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md

    :param tfrecords_dir: directory containing TFRecord files
    """

    image_count = 0
    for tfrecords_file_name in os.listdir(tfrecords_dir):

        # parse each TFRecord file
        tfrecords_file_path = os.path.join(tfrecords_dir, tfrecords_file_name)
        tf_dataset = tf.data.TFRecordDataset(tfrecords_file_path)
        for record in tf_dataset:

            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            feature = example.features.feature
            img_file_name = feature['image/filename'].bytes_list.value
            object_class_id = feature['image/object/class/text'].bytes_list.value
            x_min = feature['image/object/bbox/xmin'].float_list.value
            x_max = feature['image/object/bbox/xmax'].float_list.value
            y_min = feature['image/object/bbox/ymin'].float_list.value
            y_max = feature['image/object/bbox/ymax'].float_list.value
            width = feature['image/width'].int64_list.value[0]
            height = feature['image/height'].int64_list.value[0]

            # convert the image's bytes list back into an RGB array
            img_bytes = feature['image/encoded'].bytes_list.value[0]
            img_1d = np.fromstring(img_bytes, dtype=np.uint8)
            img = np.reshape(img_1d, (height, width, 3))

            # for each bounding box draw the rectangle and label text
            i = 0
            while i < len(x_min):

                # draw bounding box
                cv2.rectangle(img, (int(width * x_min[i]), int(height * y_min[i])),
                              (int(width * x_max[i]), int(height * y_max[i])),
                              _RECTANGLE_BGR, 2)
                # draw the label
                cv2.putText(
                    img,
                    object_class_id[i].decode(),
                    (int(width * x_min[i]) + 3, int(height * y_min[i]) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    _TEXT_BGR,
                    1,
                )
                i += 1

            # draw the annotated image
            _logger.info(f"{image_count} Displaying {len(x_min)} boxes "
                         f"for {img_file_name}")
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            image_count += 1

    cv2.destroyAllWindows()


# ------------------------------------------------------------------------------
def show_tfrecords_segmentation(
        tfrecords_dir: str,
):
    """
    Displays images with segmentation masks from all TFRecord files found in a
    directory containing TFRecord files in the format of those used for
    TensorFlow's DeepLab v3.

    The TFRecord format is assumed to contain the following feature attributes:

        'image/encoded': encoded JPG data for the base image
        'image/filename': original file name for the base image
        'image/format': 'jpeg'
        'image/height': height, in pixels, of both the base and mask images
        'image/width': width, in pixels, of both the base and mask images
        'image/channels': 3 (since the JPG is RGB data)
        'image/segmentation/class/encoded': encoded PNG data for the mask image
        'image/segmentation/class/format': 'png'

    :param tfrecords_dir: directory containing TFRecord files
    """

    image_count = 0
    for tfrecords_file_name in os.listdir(tfrecords_dir):

        # parse each TFRecord file
        tfrecords_file_path = os.path.join(tfrecords_dir, tfrecords_file_name)
        tf_dataset = tf.data.TFRecordDataset(tfrecords_file_path)
        for record in tf_dataset:

            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            feature = example.features.feature
            width = feature['image/width'].int64_list.value[0]
            height = feature['image/height'].int64_list.value[0]

            raw_img = feature['image/encoded'].bytes_list.value[0]
            img = tf.image.decode_jpeg(raw_img).numpy()
            raw_mask = feature['image/segmentation/class/encoded'].bytes_list.value[0]
            mask = tf.image.decode_png(raw_mask).numpy()

            # set the masked values to a specific pixel value
            mask[mask != 0] = 255

            # if the mask only has a single channel then convert to 3 channels
            if mask.shape[2] == 1:
                rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
                squeezed_mask = np.squeeze(mask, axis=2)
                for i in range(3):
                    rgb_mask[:, :, i] = squeezed_mask
            else:
                rgb_mask = mask

            # images are in BGR, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # overlay the mask onto the image
            alpha = 0.4
            cv2.addWeighted(rgb_mask, alpha, img, 1 - alpha, 0, img)

            # resize image
            display_width = 800
            display_height = 600
            image, _, _ = resize_with_padding(img, display_width, display_height)

            cv2.imshow("Image", image)
            cv2.waitKey(0)
            image_count += 1

    cv2.destroyAllWindows()


# ------------------------------------------------------------------------------
def bbox_coco(
        file_path: str,
) -> List[dict]:
    """
    Returns the labeled bounding boxes from a COCO annotation (*.json) file.

    :param file_path: path to a COCO annotation file
    :return: list of bounding box dictionary objects with keys "label", "x",
        "y", "w", and "h"
    """
    with open(file_path) as json_file:
        data = json.load(json_file)

    boxes = []
    for annotation in data["annotations"]:

        x, y, w, h = annotation["bbox"]
        box = {"label": "", "x": x, "y": y, "w": w, "h": h}
        boxes.append(box)

    return boxes


# ------------------------------------------------------------------------------
def bbox_darknet(
        file_path: str,
        width: int,
        height: int,
        label_indices: Dict,
) -> List[dict]:
    """
    Returns the labeled bounding boxes from a Darknet annotation (*.txt) file.

    :param file_path: path to a Darknet annotation file
    :param width: width of the corresponding image
    :param height: height of the corresponding image
    :param label_indices: dictionary mapping label indices to the corresponding
        label text
    :return: list of bounding box dictionary objects with keys "label", "x",
        "y", "w", and "h", values are percentages between 0.0 and 1.0, and the X
        and Y values are for the center of the box
    """

    boxes = []
    with open(file_path) as txt_file:

        for bbox_line in txt_file.readlines():
            parts = bbox_line.split()
            bbox_width = int(float(parts[3]) * width)
            bbox_height = int(float(parts[4]) * height)
            box = {
                # denormalize the bounding box coordinates
                "label": label_indices[int(parts[0])],
                "x": int(float(parts[1]) * width) - int(bbox_width / 2),
                "y": int(float(parts[2]) * height) - int(bbox_height / 2),
                "w": bbox_width,
                "h": bbox_height,
            }
            boxes.append(box)

    return boxes


# ------------------------------------------------------------------------------
def bbox_kitti(
        file_path: str,
        width: int,
        height: int,
) -> List[dict]:
    """
    Returns the labeled bounding boxes from a KITTI annotation (*.txt) file.

    :param file_path: path to a KITTI annotation file
    :param width: width of the corresponding image
    :param height: height of the corresponding image
    :return: list of bounding box dictionary objects with keys "label", "x",
        "y", "w", and "h"
    """

    boxes = []
    with open(file_path) as txt_file:

        for bbox_line in txt_file.readlines():

            parts = bbox_line.split()
            label = parts[0]
            bbox_min_x = int(float(parts[4]))
            bbox_min_y = int(float(parts[5]))
            bbox_max_x = int(float(parts[6]))
            bbox_max_y = int(float(parts[7]))

            # perform sanity checks on max values, failures won't cause a skip
            if bbox_max_x >= width:
                # report the issue via log message
                _logger.warning(
                    "Bounding box maximum X is greater than width in KITTI "
                    f"annotation file {file_path}",
                )

                # fix the issue
                bbox_max_x = width - 1

            if bbox_max_y >= height:
                # report the issue via log message
                _logger.warning(
                    "Bounding box maximum Y is greater than height in KITTI "
                    f"annotation file {file_path}",
                )

                # fix the issue
                bbox_max_y = height - 1

            # get the box's dimensions
            bbox_width = bbox_max_x - bbox_min_x + 1
            bbox_height = bbox_max_y - bbox_min_y + 1

            # make sure we don't have wonky values with mins > maxs
            if (bbox_min_x >= bbox_max_x) or (bbox_min_y >= bbox_max_y):
                # report the issue via log message
                _logger.warning(
                    "Bounding box minimum(s) greater than maximum(s) in KITTI "
                    f"annotation file {file_path} -- skipping",
                )

                # skip this box since it's not clear how to fix it
                continue

            # include this box in the list we'll return
            box = {
                "label": label,
                "x": bbox_min_x,
                "y": bbox_min_y,
                "w": bbox_width,
                "h": bbox_height,
            }
            boxes.append(box)

    return boxes


# ------------------------------------------------------------------------------
def bbox_pascal(
        file_path: str,
        width: int,
        height: int,
) -> List[dict]:
    """
    Returns the labeled bounding boxes from a PASCAL VOC annotation (*.xml) file.

    :param file_path: path to a PASCAL VOC annotation file
    :param width: width of the corresponding image
    :param height: height of the corresponding image
    :return: list of bounding box dictionary objects with keys "label", "x",
        "y", "w", and "h"
    """

    boxes = []

    # load the contents of the annotations file into an ElementTree
    tree = ElementTree.parse(file_path)

    # sanity check on the expected image dimensions
    size = tree.find("size")
    expected_width = int(size.find("width").text)
    expected_height = int(size.find("height").text)
    if expected_width != width:
        raise ValueError(
            f"File: {file_path}\nUnexpected width of image -- "
            f"expected: {expected_width}, actual: {width}",
        )
    if expected_height != height:
        raise ValueError(
            f"File: {file_path}\nUnexpected height of image -- "
            f"expected: {expected_height}, actual: {height}",
        )

    for obj in tree.iter("object"):

        label = obj.find("name").text
        bndbox = obj.find("bndbox")
        bbox_min_x = int(float(bndbox.find("xmin").text))
        bbox_min_y = int(float(bndbox.find("ymin").text))
        bbox_max_x = int(float(bndbox.find("xmax").text))
        bbox_max_y = int(float(bndbox.find("ymax").text))

        # perform sanity checks on max values, failures won't cause a skip
        if bbox_max_x >= width:
            # report the issue via log message
            _logger.warning("Bounding box maximum X is greater than width")

            # fix the issue
            bbox_max_x = width - 1

        if bbox_max_y >= height:
            # report the issue via log message
            _logger.warning("Bounding box maximum Y is greater than height")

            # fix the issue
            bbox_max_y = height - 1

        # get the box's dimensions
        bbox_width = bbox_max_x - bbox_min_x + 1
        bbox_height = bbox_max_y - bbox_min_y + 1

        # make sure we don't have wonky values with mins > maxs
        if (bbox_min_x >= bbox_max_x) or (bbox_min_y >= bbox_max_y):

            # report the issue via log message
            _logger.warning("Bounding box minimum(s) greater than maximum(s)")

            # skip this box since it's not clear how to fix it
            continue

        # include this box in the list we'll return
        box = {
            "label": label,
            "x": bbox_min_x,
            "y": bbox_min_y,
            "w": bbox_width,
            "h": bbox_height,
        }
        boxes.append(box)

    return boxes


# ------------------------------------------------------------------------------
def main():

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--annotations",
        required=True,
        type=str,
        help="annotations directory path",
    )
    args_parser.add_argument(
        "--images",
        required=False,
        type=str,
        help="images directory path",
    )
    args_parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=FORMAT_CHOICES,
        help="annotation format",
    )
    args_parser.add_argument(
        "--darknet_labels",
        type=str,
        required=False,
        help="labels file for label indices used in Darknet annotations",
    )
    args = vars(args_parser.parse_args())

    if args["format"] == "openimages":

        # read the OpenImages CSV into a pandas DataFrame
        df_annotations = pd.read_csv(args["annotations"])

        count = 0
        for image_file_name in os.listdir(args["images"]):

            count += 1
            image_id = os.path.splitext(image_file_name)[0]
            bboxes = df_annotations.loc[df_annotations["ImageID"] == image_id]
            if len(bboxes) > 0:
                image = cv2.imread(os.path.join(args["images"], image_file_name))
                for _, bbox in bboxes.iterrows():
                    # draw the box
                    cv2.rectangle(
                        image,
                        (bbox["XMin"], bbox["YMin"]),
                        (bbox["XMax"], bbox["YMax"]),
                        _RECTANGLE_BGR,
                        2,
                    )
                    # draw the label
                    cv2.putText(
                        image,
                        bbox["ClassName"],
                        (bbox["XMin"] + 3, bbox["YMin"] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        _TEXT_BGR,
                        1,
                    )
                _logger.info(f"{count} Displaying {len(bboxes)} boxes for {image_file_name}")

                # show the output image
                cv2.imshow("Image", image)
                cv2.waitKey(0)

    elif args["format"] == "tfrecord":

        # use the display function for TFRecords with segmentation (masks)
        show_tfrecords_segmentation(args["annotations"])

    else:

        # get the label indices for Darknet
        darknet_label_indices = {}
        if args["format"] == "darknet":
            with open(args["darknet_labels"], "r") as darknet_labels_file:
                label_index = 0
                for line in darknet_labels_file:
                    darknet_label_indices[label_index] = line.strip()
                    label_index += 1

        # for each annotation file we'll draw all bounding boxes on the corresponding image
        count = 0
        for annotation_file_name in os.listdir(args["annotations"]):

            count += 1

            # we assume each annotation shares the same base name as the corresponding image
            annotations_file_path = \
                os.path.join(args["annotations"], annotation_file_name)
            image_file_name = os.path.splitext(annotation_file_name)[0] + ".jpg"

            # load the input image from disk to determine the dimensions
            image_file_path = \
                os.path.join(args["images"], image_file_name)
            image = cv2.imread(image_file_path)
            try:
                image_height, image_width = image.shape[:2]
            except AttributeError as attribute_error:
                _logger.error(f"Bad image: {image_file_path}", attribute_error)
                continue

            # read the bounding boxes from the annotation file
            # bboxes = []

            if args["format"] == "coco":
                if annotation_file_name.endswith(".json"):
                    bboxes = bbox_coco(annotations_file_path)
                else:
                    continue
            elif args["format"] == "darknet":
                if annotation_file_name.endswith(".txt"):
                    bboxes = bbox_darknet(annotations_file_path, image_width, image_height, darknet_label_indices)
                else:
                    continue
            elif args["format"] == "kitti":
                if annotation_file_name.endswith(".txt"):
                    bboxes = bbox_kitti(annotations_file_path, image_width, image_height)
                else:
                    continue
            elif args["format"] == "pascal":
                if annotation_file_name.endswith(".xml"):
                    try:
                        bboxes = bbox_pascal(annotations_file_path, image_width, image_height)
                    except ValueError as value_error:
                        _logger.error(f"Bad annotation: {annotations_file_path}", value_error)
                        continue
                else:
                    continue
            else:
                raise ValueError(f"Invalid format argument: \'{args['format']}\'")

            # for each bounding box draw the rectangle
            for bbox in bboxes:

                # draw the box
                cv2.rectangle(
                    image,
                    (bbox["x"], bbox["y"]),
                    (bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]),
                    _RECTANGLE_BGR,
                    2,
                )
                # draw the label
                cv2.putText(
                    image,
                    bbox["label"],
                    (bbox["x"] + 3, bbox["y"] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    _TEXT_BGR,
                    1,
                )
            _logger.info(f"{count} Displaying {len(bboxes)} boxes for {image_file_name}")

            # show the output image
            cv2.imshow("Image", image)
            cv2.waitKey(0)


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # Visualize images with bounding boxes specified by corresponding annotations.
    #
    # Usage:
    #
    # $ python <this_script.py> --images /home/ubuntu/data/handgun/images \
    #       --annotations /home/ubuntu/data/handgun/annotations/coco \
    #       --format coco
    #
    # $ python <this_script.py> --images /nvidia/tlt/experiments/kitti/training/image_2 \
    #       --annotations /nvidia/tlt/experiments/tfrecords/training \
    #       --format tfrecord

    main()
