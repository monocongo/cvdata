import logging
import os

import cv2
import pytest
# scikit-image version <0.16
from skimage.measure import compare_mse as mean_squared_error
# scikit-image version >=0.16
# from skimage.metrics import mean_squared_error

from cvdata import convert
from tests.assert_utils import images_equal


# ------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)


# ------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "data_dir",
)
def test_png_to_jpg(
        data_dir,
):
    """
    Test for the cvdata.convert.png_to_jpg() function

    :param data_dir: temporary directory into which test files will be loaded
    """
    png_file_path = os.path.join(str(data_dir), "james.png")
    jpg_file_path = convert.png_to_jpg(png_file_path)
    converted_jpg_file_path = os.path.join(str(data_dir), "james.jpg")
    assert jpg_file_path == converted_jpg_file_path
    expected_jpg_file_path = os.path.join(str(data_dir), "expected_james.jpg")
    assert images_equal(converted_jpg_file_path, expected_jpg_file_path)
