import os
from typing import Set

import PIL.Image as Image


# ------------------------------------------------------------------------------
def image_dimensions(
        image_file_path: str,
) -> (int, int, int):
    """
    Gets an image's width, height, and depth dimensions.

    :param image_file_path: absolute path to an image file
    :return: the image's width, height, and depth
    """

    image = Image.open(image_file_path)
    img_width, img_height = image.size
    if image_file_path.lower().endswith("png"):
        img_depth = 1
    else:
        img_depth = image.layers
    return img_width, img_height, img_depth


# ------------------------------------------------------------------------------
def matching_ids(
        annotations_dir: str,
        images_dir: str,
        annotations_ext: str,
        images_ext: str,
) -> Set[str]:
    """
    Given a directory and extension to use for image files and annotation files,
    find the matching file IDs across the two directories. Useful to find
    matching image and annotation files.

    For example, a match would be where we have two files
    <image_dir>/<file_id><images_ext> and <annotations_dir>/<file_id><annotations_ext>
    if <file_id> is the same for both files.

    :param annotations_dir:
    :param images_dir:
    :param annotations_ext:
    :param images_ext:
    :return:
    """

    # define a function to get all file IDs in a directory
    # where the file has the specified extension
    def file_ids(directory: str, extension: str):
        ids = []
        for file_name in os.listdir(directory):
            file_id, ext = os.path.splitext(file_name)
            if ext == extension:
                ids.append(file_id)
        return ids

    # get the list of file IDs matching to the relevant extensions
    ids_annotations = file_ids(annotations_dir, annotations_ext)
    ids_image = file_ids(images_dir, images_ext)

    # return the set of file IDs in common for the annotations and images
    return set(ids_annotations) & set(ids_image)
