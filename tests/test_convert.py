import logging
import os

import pytest

from cvdata import convert

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
    expected_jpg_file_path = os.path.join(str(data_dir), "expected_james.jpg")
    assert jpg_file_path == expected_jpg_file_path
