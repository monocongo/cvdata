import logging
from xml.etree import ElementTree

import pytest

from cvdata import rename

# ------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)


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
    rename.relabel_pascal(pascal_file_path, "NOT_PRESENT", "NOT_USED")
    etree_after_relabel = ElementTree.parse(pascal_file_path)
    assert elements_equal(
        etree_before_relabel.getroot(),
        etree_after_relabel.getroot(),
    )

    # confirm that relabeling occurred as expected
    rename.relabel_pascal(pascal_file_path, "pistol", "firearm")
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
        rename.relabel_pascal(None, "don't care", "don't care")
        rename.relabel_pascal(pascal_file_path, None, "don't care")
        rename.relabel_pascal(pascal_file_path, "don't care", None)
        rename.relabel_pascal("/not/present", "don't care", "don't care")
        rename.relabel_pascal(1, "don't care", "don't care")
        rename.relabel_pascal(1.0, "don't care", "don't care")
        rename.relabel_pascal(pascal_file_path, 1, "don't care")
        rename.relabel_pascal(pascal_file_path, 1.0, "don't care")
        rename.relabel_pascal(pascal_file_path, True, "don't care")
        rename.relabel_pascal(pascal_file_path, "don't care", 1)
        rename.relabel_pascal(pascal_file_path, "don't care", 1.0)
        rename.relabel_pascal(pascal_file_path, "don't care", True)


def elements_equal(e1, e2):
    if e1.tag != e2.tag: return False
    if e1.text != e2.text: return False
    if e1.tail != e2.tail: return False
    if e1.attrib != e2.attrib: return False
    if len(e1) != len(e2): return False
    return all(elements_equal(c1, c2) for c1, c2 in zip(e1, e2))
