import logging
from xml.etree import ElementTree

import pytest

from cvdata import relabel
from assert_utils import elements_equal, text_files_equal

# ------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)


# ------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "data_dir",
)
def test_relabel_darknet(
        data_dir,
):
    """
    Test for the cvdata.relabel.relabel_darknet() function

    :param data_dir: temporary directory into which test files will be loaded
    """
    darknet_file_name = "darknet_1.txt"
    darknet_file_path = str(data_dir.join(darknet_file_name))

    # confirm that a relabeling won't occur if the old value is not present
    relabel.relabel_darknet(darknet_file_path, 58, 59)
    expected_darknet_file_name = "expected_darknet_1.txt"
    expected_darknet_file_path = str(data_dir.join(expected_darknet_file_name))
    assert text_files_equal(
        darknet_file_path,
        expected_darknet_file_path,
    )

    # confirm that relabeling occurred as expected
    relabel.relabel_darknet(darknet_file_path, 3, 2)
    expected_darknet_file_name = "expected_darknet_2.txt"
    expected_darknet_file_path = str(data_dir.join(expected_darknet_file_name))
    assert text_files_equal(
        darknet_file_path,
        expected_darknet_file_path,
    )

    # confirm that various invalid arguments raise an error
    with pytest.raises(TypeError):
        relabel.relabel_darknet(darknet_file_path, None, 0)
        relabel.relabel_darknet(darknet_file_path, 0, None)
        relabel.relabel_darknet(1, 0, 1)
        relabel.relabel_darknet(None, 1, 0)
        relabel.relabel_darknet("/not/present", 0, 1)
        relabel.relabel_darknet(1.0, "strings won't work", 0)
        relabel.relabel_darknet(darknet_file_path, 1, "strings won't work")
        relabel.relabel_darknet(darknet_file_path, 1.0, 0)
        relabel.relabel_darknet(darknet_file_path, 2, 1.0)
        relabel.relabel_darknet(darknet_file_path, True, 0)
        relabel.relabel_darknet(darknet_file_path, 1, True)
    with pytest.raises(ValueError):
        relabel.relabel_darknet(darknet_file_path, -5, 1)
        relabel.relabel_darknet(darknet_file_path, 1, -4)


# ------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "data_dir",
)
def test_relabel_kitti(
        data_dir,
):
    """
    Test for the cvdata.relabel.relabel_kitti() function

    :param data_dir: temporary directory into which test files will be loaded
    """
    kitti_file_name = "kitti_1.txt"
    kitti_file_path = str(data_dir.join(kitti_file_name))

    # confirm that a relabeling won't occur if the old value is not present
    relabel.relabel_kitti(kitti_file_path, "NOT_PRESENT", "NOT_USED")
    expected_kitti_file_name = "expected_kitti_1.txt"
    expected_kitti_file_path = str(data_dir.join(expected_kitti_file_name))
    assert text_files_equal(
        kitti_file_path,
        expected_kitti_file_path,
    )

    # confirm that relabeling occurred as expected
    relabel.relabel_kitti(kitti_file_path, "pistol", "firearm")
    expected_kitti_file_name = "expected_kitti_2.txt"
    expected_kitti_file_path = str(data_dir.join(expected_kitti_file_name))
    assert text_files_equal(
        kitti_file_path,
        expected_kitti_file_path,
    )

    # confirm that various invalid arguments raise an error
    with pytest.raises(ValueError):
        relabel.relabel_kitti(None, "don't care", "don't care")
        relabel.relabel_kitti(kitti_file_path, None, "don't care")
        relabel.relabel_kitti(kitti_file_path, "don't care", None)
        relabel.relabel_kitti("/not/present", "don't care", "don't care")
        relabel.relabel_kitti(1, "don't care", "don't care")
        relabel.relabel_kitti(1.0, "don't care", "don't care")
        relabel.relabel_kitti(kitti_file_path, 1, "don't care")
        relabel.relabel_kitti(kitti_file_path, 1.0, "don't care")
        relabel.relabel_kitti(kitti_file_path, True, "don't care")
        relabel.relabel_kitti(kitti_file_path, "don't care", 1)
        relabel.relabel_kitti(kitti_file_path, "don't care", 1.0)
        relabel.relabel_kitti(kitti_file_path, "don't care", True)


# ------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "data_dir",
)
def test_relabel_pascal(
        data_dir,
):
    """
    Test for the cvdata.relabel.relabel_pascal() function

    :param data_dir: temporary directory into which test files will be loaded
    """
    pascal_file_name = "pascal_1.xml"
    pascal_file_path = str(data_dir.join(pascal_file_name))

    # confirm that a relabeling won't occur if the old value is not present
    etree_before_relabel = ElementTree.parse(pascal_file_path)
    relabel.relabel_pascal(pascal_file_path, "NOT_PRESENT", "NOT_USED")
    etree_after_relabel = ElementTree.parse(pascal_file_path)
    assert elements_equal(
        etree_before_relabel.getroot(),
        etree_after_relabel.getroot(),
    )

    # confirm that relabeling occurred as expected
    relabel.relabel_pascal(pascal_file_path, "pistol", "firearm")
    etree_after_relabel = ElementTree.parse(pascal_file_path)
    expected_pascal_file_name = "expected_pascal_1.xml"
    expected_pascal_file_path = str(data_dir.join(expected_pascal_file_name))
    etree_expected_after_relabel = ElementTree.parse(expected_pascal_file_path)
    assert elements_equal(
        etree_expected_after_relabel.getroot(),
        etree_after_relabel.getroot(),
    )

    # confirm that various invalid arguments raise an error
    with pytest.raises(ValueError):
        relabel.relabel_pascal(None, "don't care", "don't care")
        relabel.relabel_pascal(pascal_file_path, None, "don't care")
        relabel.relabel_pascal(pascal_file_path, "don't care", None)
        relabel.relabel_pascal("/not/present", "don't care", "don't care")
        relabel.relabel_pascal(1, "don't care", "don't care")
        relabel.relabel_pascal(1.0, "don't care", "don't care")
        relabel.relabel_pascal(pascal_file_path, 1, "don't care")
        relabel.relabel_pascal(pascal_file_path, 1.0, "don't care")
        relabel.relabel_pascal(pascal_file_path, True, "don't care")
        relabel.relabel_pascal(pascal_file_path, "don't care", 1)
        relabel.relabel_pascal(pascal_file_path, "don't care", 1.0)
        relabel.relabel_pascal(pascal_file_path, "don't care", True)
