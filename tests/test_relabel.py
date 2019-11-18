import logging
from xml.etree import ElementTree

import pytest

from cvdata import relabel

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
    with pytest.raises(ValueError) as ex_info:
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


# ------------------------------------------------------------------------------
def elements_equal(e1, e2):
    """
    Utility function to compare Element objects.

    From https://stackoverflow.com/a/24349916/85248

    :param e1:
    :param e2:
    :return: True if equal, False if not
    """

    if e1.tag != e2.tag:
        return False
    if e1.text != e2.text:
        return False
    if e1.tail != e2.tail:
        return False
    if e1.attrib != e2.attrib:
        return False
    if len(e1) != len(e2):
        return False
    return all(elements_equal(c1, c2) for c1, c2 in zip(e1, e2))


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
