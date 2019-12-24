import logging
import os

import cv2
import numpy as np
import pytest

from cvdata import resize
from assert_utils import text_files_equal, xml_equal

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
    Test for the cvdata.resize.resize() function

    :param data_dir: temporary directory into which test files will be loaded
    """
    file_id = "image"
    image_ext = ".jpg"
    image_file_name = f"{file_id}{image_ext}"
    kitti_ext = ".txt"
    kitti_file_name = f"{file_id}{kitti_ext}"
    pascal_ext = ".xml"
    pascal_file_name = f"{file_id}{pascal_ext}"

    # make a directory to hold our resized files
    resized_dir = os.path.join(str(data_dir), "resized")
    os.makedirs(resized_dir, exist_ok=True)

    new_width = 240
    new_height = 720
    expected_resized_file_id = f"{file_id}_w{new_width}_h{new_height}"
    expected_resized_image_file_name = f"{expected_resized_file_id}{image_ext}"
    expected_resized_kitti_file_name = f"{expected_resized_file_id}{kitti_ext}"
    expected_resized_pascal_file_name = f"{expected_resized_file_id}{pascal_ext}"
    expected_resized_image_file_path = os.path.join(str(data_dir), expected_resized_image_file_name)
    expected_resized_kitti_file_path = os.path.join(str(data_dir), expected_resized_kitti_file_name)
    expected_resized_pascal_file_path = os.path.join(str(data_dir), expected_resized_pascal_file_name)

    # confirm that a resizing won't occur if the expected files are not present
    # TODO

    # confirm that resizing occurred as expected for a KITTI annotated image
    resize.resize_image(
        file_id + image_ext,
        data_dir,
        resized_dir,
        new_width,
        new_height,
    )
    resized_image_file_path = os.path.join(resized_dir, image_file_name)
    resized_image = cv2.imread(resized_image_file_path)
    expected_resized_image = cv2.imread(expected_resized_image_file_path)
    np.testing.assert_equal(resized_image,
                            expected_resized_image,
                            err_msg="Image not resized as expected")
    resized_kitti_file_path = os.path.join(resized_dir, kitti_file_name)
    assert text_files_equal(resized_kitti_file_path, expected_resized_kitti_file_path)

    # confirm that resizing occurred as expected for a PASCAL annotated image
    resize.resize_image(
        file_id + image_ext,
        data_dir,
        resized_dir,
        new_width,
        new_height,
    )
    resized_image_file_path = os.path.join(resized_dir, image_file_name)
    resized_image = cv2.imread(resized_image_file_path)
    expected_resized_image = cv2.imread(expected_resized_image_file_path)
    np.testing.assert_equal(resized_image,
                            expected_resized_image,
                            err_msg="Image not resized as expected")
    resized_pascal_file_path = os.path.join(resized_dir, pascal_file_name)
    assert xml_equal(resized_pascal_file_path, expected_resized_pascal_file_path)
