import logging
import os

import cv2
import numpy as np
import pytest

from cvdata import resize

# ------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)


# ------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "data_dir",
)
def test_resize(
        data_dir,
):
    """
    Test for the cvdata.resize() function

    :param data_dir: temporary directory into which test files will be loaded
    """
    file_id = "image"
    image_ext = ".jpg"
    image_file_name = f"{file_id}{image_ext}"
    input_image_file_path = str(data_dir.join(image_file_name))
    kitti_ext = ".txt"
    kitti_file_name = f"{file_id}{kitti_ext}"
    kitti_file_path = str(data_dir.join(kitti_file_name))
    pascal_ext = ".xml"
    pascal_file_name = f"{file_id}{pascal_ext}"
    pascal_file_path = str(data_dir.join(pascal_file_name))

    # make a directory to hold our resized files
    resized_dir = os.path.join(str(data_dir), "resized")
    os.makedirs(resized_dir, exist_ok=True)

    new_width = 240
    new_height = 720
    resized_file_id = f"{file_id}_w{new_width}_h{new_height}"
    resized_image_file_name = f"{resized_file_id}{image_ext}"
    resized_image_file_path = os.path.join(resized_dir, image_file_name)
    expected_resized_image_file_path = os.path.join(resized_dir, resized_image_file_name)
    resized_kitti_file_name = f"{resized_file_id}{kitti_ext}"
    resized_kitti_file_path = os.path.join(resized_dir, kitti_file_name)
    expected_resized_kitti_file_path = os.path.join(resized_dir, resized_kitti_file_name)
    resized_pascal_file_name = f"{resized_file_id}{pascal_ext}"
    resized_pascal_file_path = os.path.join(resized_dir, pascal_file_name)
    expected_resized_pascal_file_path = os.path.join(resized_dir, resized_pascal_file_name)

    # confirm that a resizing won't occur if the expected files are not present
    # TODO

    # confirm that resizing occurred as expected
    resize.resize_image(file_id, image_ext, pascal_ext, data_dir, data_dir, resized_dir, resized_dir, new_width, new_height, "pascal")
    resize.resize_image(file_id, image_ext, kitti_ext, data_dir, data_dir, resized_dir, resized_dir, new_width, new_height, "kitti")

    resized_image = cv2.imread(os.path.join(resized_dir, image_file_name))
    expected_resized_image = cv2.imread(resized_image_file_path)
    np.testing.assert_equal(resized_image,
                            expected_resized_image,
                            err_msg="Image not resized as expected")


# ------------------------------------------------------------------------------
def text_files_equal(
        file_path_1: str,
        file_path_2: str,
):
    """
    Utility function to compare Element objects.

    :param file_path_1:
    :param file_path_2:
    :return: True if equal, False if not
    """
    with open(file_path_1, "r") as file_1, \
            open(file_path_2, "r") as file_2:
        for line_1, line_2 in zip(file_1.readline(), file_2.readline()):
            if line_1 != line_2:
                return False

    return True
