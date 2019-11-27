import logging
import os
import shutil
from typing import Dict

from lxml import etree
from PIL import Image

from cvdata.convert import png_to_jpg
from cvdata.utils import matching_ids


# ------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
)
_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def clean_pascal(
        pascal_dir: str,
        images_dir: str,
        rename_labels: Dict,
        problems_dir: str = None,
):

    # convert all PNG images to JPG, and remove the original PNG file
    for file_id in matching_ids(pascal_dir, images_dir, ".xml", ".png"):
        png_file_path = os.path.join(images_dir, file_id + ".png")
        png_to_jpg(png_file_path, remove_png=True)

    # get a set of file IDs of the PASCAL VOC annotations and corresponding images
    file_ids = matching_ids(pascal_dir, images_dir, ".xml", ".jpg")

    # make the problem files directory if necessary, in case it doesn't already exist
    if problems_dir is not None:
        os.makedirs(problems_dir, exist_ok=True)

    # remove files that aren't matches
    for dir in [pascal_dir, images_dir]:
        for file_name in os.listdir(dir):
            # only filter out image and PASCAL files (this is needed
            # in case a subdirectory exists in the directory)
            if file_name.endswith(".xml") or file_name.endswith(".jpg"):
                if os.path.splitext(file_name)[0] not in file_ids:
                    unmatched_file = os.path.join(dir, file_name)
                    if problems_dir is not None:
                        shutil.move(unmatched_file, os.path.join(problems_dir, file_name))
                    else:
                        os.remove(unmatched_file)

    # loop over all the matching files and clean the PASCAL annotations
    for i, file_id in enumerate(file_ids):

        # get the image width and height
        jpg_file_name = file_id + ".jpg"
        image_file_path = os.path.join(images_dir, jpg_file_name)
        image = Image.open(image_file_path)
        img_width, img_height = image.size

        # update the image file name in the PASCAL file
        src_annotation_file_path = os.path.join(pascal_dir, file_id + ".xml")
        if os.path.exists(src_annotation_file_path):
            tree = etree.parse(src_annotation_file_path)
            root = tree.getroot()

            size = tree.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)

            if (width != img_width) or (height != img_height):
                # something's amiss that we can't reasonably fix, remove files
                if problems_dir is not None:
                    for file_path in [src_annotation_file_path, image_file_path]:
                        dest_file_path = os.path.join(problems_dir, os.path.split(file_path)[1])
                        shutil.move(file_path, dest_file_path)
                else:
                    os.remove(src_annotation_file_path)
                    os.remove(image_file_path)
                continue

            # update the image file name
            file_name = root.find("filename")
            if (file_name is not None) and (file_name != jpg_file_name):
                file_name.text = jpg_file_name

            # loop over all bounding boxes
            for obj in root.iter("object"):

                # rename all bounding box labels if specified in the rename dictionary
                name = obj.find("name")
                if name is None:
                    # drop the bounding box since it is useless with no label
                    parent = obj.getparent()
                    parent.remove(obj)
                elif name.text in rename_labels:
                    # update the label
                    name.text = rename_labels[name.text]

                # for each bounding box make sure we have max
                # values that are one less than the width/height
                bbox = obj.find("bndbox")
                bbox_min_x = int(float(bbox.find("xmin").text))
                bbox_min_y = int(float(bbox.find("ymin").text))
                bbox_max_x = int(float(bbox.find("xmax").text))
                bbox_max_y = int(float(bbox.find("ymax").text))

                # make sure we don't have wonky values with mins > maxs
                if (bbox_min_x >= bbox_max_x) or (bbox_min_y >= bbox_max_y):
                    # drop the bounding box
                    _logger.warning(
                        "Dropping bounding box for object in file "
                        f"{src_annotation_file_path} due to invalid "
                        "min/max values",
                    )
                    parent = obj.getparent()
                    parent.remove(obj)

                else:
                    # make sure the max values don't go past the edge
                    if bbox_max_x >= img_width:
                        bbox.find("xmax").text = str(img_width - 1)
                    if bbox_max_y >= img_height:
                        bbox.find("ymax").text = str(img_height - 1)

            # drop the image path, it's not reliable
            path = root.find("path")
            if path is not None:
                parent = path.getparent()
                parent.remove(path)

            # drop the image folder, it's not reliable
            folder = root.find("folder")
            if folder is not None:
                parent = folder.getparent()
                parent.remove(folder)

            # write the tree back to file
            tree.write(src_annotation_file_path)
