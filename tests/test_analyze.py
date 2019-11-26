import logging
import os

import pytest

from cvdata import analyze

# ------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)


# ------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "data_dir",
)
def test_count_labels(
        data_dir,
):
    """
    Test for the cvdata.analyze.count_labels() function

    :param data_dir: temporary directory into which test files will be loaded
    """
    annotation_format = "kitti"
    kitti_file_path = os.path.join(str(data_dir), annotation_format, "kitti_1.txt")
    label_counts = analyze.count_labels(kitti_file_path, annotation_format)
    assert label_counts["person"] == 4
    assert label_counts["truck"] == 1
    assert label_counts["car"] == 1
