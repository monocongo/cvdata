from xml.etree import ElementTree


# ------------------------------------------------------------------------------
def elements_equal(e1, e2) -> bool:

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
def xml_equal(
        xml_file_path_1: str,
        xml_file_path_2: str,
) -> bool:
    """
    Utility function to compare XML files.

    From https://stackoverflow.com/a/24349916/85248

    :param xml_file_path_1: path to an XML file
    :param xml_file_path_2: path to an XML file
    :return: True if XML contents are equal, False if not
    """

    root_1 = ElementTree.parse(xml_file_path_1).getroot()
    root_2 = ElementTree.parse(xml_file_path_2).getroot()

    return elements_equal(root_1, root_2)


# ------------------------------------------------------------------------------
def text_files_equal(
        file_path_1: str,
        file_path_2: str,
) -> bool:
    """
    Utility function to compare text files.

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
