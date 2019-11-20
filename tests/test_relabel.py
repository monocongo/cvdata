import logging
from xml.etree import ElementTree

import pytest

from cvdata import relabel
from tests.assert_utils import elements_equal, text_files_equal

# ------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)


# ------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "data_dir",
)
def test_relabel_kitti(
        data_dir,
):
    """
    Test for the cvdata.relabel_kitti() function

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

    # confirm that invalid argument types raise an error
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
    Test for the cvdata.relabel_pascal() function

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

    # confirm that invalid argument types raise an error
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
