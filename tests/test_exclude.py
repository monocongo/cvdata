import logging
import os

import pandas as pd
import pytest

from cvdata import exclude

# ------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)


# ------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "data_dir",
)
def test_exclude_files(
        data_dir,
):
    """
    Test for the cvdata.exclude.exclude_files() function

    :param data_dir: temporary directory into which test files will be loaded
    """
    exclusions_file_path = os.path.join(str(data_dir), "exclusions.txt")
    darknet_dir = os.path.join(str(data_dir), "darknet")
    darknet_images_dir = os.path.join(darknet_dir, "images")
    kitti_dir = os.path.join(str(data_dir), "kitti")
    kitti_images_dir = os.path.join(kitti_dir, "images")
    openimages_dir = os.path.join(str(data_dir), "openimages")
    openimages_images_dir = os.path.join(openimages_dir, "images")
    openimages_csv = os.path.join(openimages_dir, "openimages.csv")
    pascal_dir = os.path.join(str(data_dir), "pascal")
    pascal_images_dir = os.path.join(pascal_dir, "images")

    exclude.exclude_files(exclusions_file_path, darknet_images_dir, darknet_dir, "darknet")
    assert set(os.listdir(darknet_dir)) == set(["keep_1.txt", "images"])
    assert set(os.listdir(darknet_images_dir)) == set(["keep_1.jpg"])
    exclude.exclude_files(exclusions_file_path, kitti_images_dir, kitti_dir, "kitti")
    assert set(os.listdir(kitti_dir)) == set(["keep_1.txt", "images"])
    assert set(os.listdir(kitti_images_dir)) == set(["keep_1.jpg"])
    exclude.exclude_files(exclusions_file_path, openimages_images_dir, openimages_csv, "openimages")
    assert set(os.listdir(openimages_dir)) == set(["openimages.csv", "images"])
    df = pd.read_csv(openimages_csv)
    assert df["ImageID"][0] == "4fa8e06fc7b99ae4"
    exclude.exclude_files(exclusions_file_path, pascal_images_dir, pascal_dir, "pascal")
    assert set(os.listdir(pascal_dir)) == set(["keep_1.xml", "images"])
    assert set(os.listdir(pascal_images_dir)) == set(["keep_1.jpg"])

    # confirm that various invalid arguments raise an error
    with pytest.raises(ValueError):
        exclude.exclude_files(exclusions_file_path, pascal_images_dir, pascal_dir, "unknown")
        exclude.exclude_files(exclusions_file_path, pascal_images_dir, None, "pascal")
        exclude.exclude_files(exclusions_file_path, None, pascal_dir, "pascal")
        exclude.exclude_files(None, pascal_images_dir, pascal_dir, "pascal")
        exclude.exclude_files("unknown", pascal_images_dir, pascal_dir, "pascal")
