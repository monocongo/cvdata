import logging
import os

import pytest

from cvdata import clean
from assert_utils import images_equal, text_files_equal, xml_equal

# ------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)


# ------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "data_dir",
)
def test_clean_kitti(
        data_dir,
):
    """
    Test for the cvdata.clean.clean_kitti() function

    :param data_dir: temporary directory into which test files will be loaded
    """
    problems_dir = os.path.join(str(data_dir), 'problems')
    clean.clean_kitti(
        str(data_dir), 
        str(data_dir), 
        {"pistol": "handgun"}, 
        label_keep=["handgun", "weapon"],
        problems_dir=problems_dir
    )

    # make sure that the test expected JPG image and KITTI label
    # files were correctly moved into the "problems" directory
    expected_image_path = os.path.join(problems_dir, 'converted_handgun_00941.jpg')
    assert os.path.exists(expected_image_path)
    expected_kitti_path_1 = os.path.join(problems_dir, 'expected_handgun_00941.txt')
    assert os.path.exists(expected_kitti_path_1)
    expected_kitti_path_2 = os.path.join(problems_dir, 'expected_handgun_00929.txt')
    assert os.path.exists(expected_kitti_path_2)

    # make sure that the original PNG image was converted to JPG
    converted_image_path = os.path.join(str(data_dir), 'handgun_00941.jpg')
    assert os.path.exists(converted_image_path)

    # make sure the original PNG file was deleted
    assert not os.path.exists(os.path.join(str(data_dir), 'handgun_00941.png'))

    # make sure that the image was correctly converted to JPG
    assert images_equal(converted_image_path, expected_image_path)

    # make sure that the KITTI file was correctly updated
    assert text_files_equal(expected_kitti_path_1, os.path.join(data_dir, 'handgun_00941.txt'))
    assert text_files_equal(expected_kitti_path_2, os.path.join(data_dir, 'handgun_00929.txt'))


# ------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "data_dir",
)
def test_clean_pascal(
        data_dir,
):
    """
    Test for the cvdata.clean.clean_pascal() function

    :param data_dir: temporary directory into which test files will be loaded
    """
    problems_dir = os.path.join(str(data_dir), 'problems')
    clean.clean_pascal(
        str(data_dir), 
        str(data_dir), 
        {"Hammer": "hammer"}, 
        label_keep=["Hammer"],
        problems_dir=problems_dir
    )

    # make sure that the original PNG image was converted to JPG
    converted_image_path = os.path.join(str(data_dir), 'beeeab1b6768f8fc.jpg')
    assert os.path.exists(converted_image_path)

    # make sure the original PNG file was deleted
    assert not os.path.exists(os.path.join(str(data_dir), 'beeeab1b6768f8fc.png'))

    # make sure that the test expected JPG image and PASCAL
    # files were correctly moved into the "problems" directory
    expected_image_path = os.path.join(problems_dir, 'converted_beeeab1b6768f8fc.jpg')
    assert os.path.exists(expected_image_path)
    expected_pascal_path = os.path.join(problems_dir, 'expected_beeeab1b6768f8fc.xml')
    assert os.path.exists(expected_pascal_path)

    # make sure that the image was correctly converted to JPG
    expected_image_path = os.path.join(problems_dir, 'converted_beeeab1b6768f8fc.jpg')
    assert os.path.exists(expected_image_path)
    assert images_equal(converted_image_path, expected_image_path)

    # make sure that the PASCAL file was correctly updated
    assert xml_equal(
        expected_pascal_path,
        os.path.join(str(data_dir), 'beeeab1b6768f8fc.xml'),
    )
