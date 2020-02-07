![build](https://github.com/monocongo/cvdata/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/monocongo/cvdata/branch/master/graph/badge.svg)](https://codecov.io/gh/monocongo/cvdata)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cvdata)
<!-- [![CircleCI](https://circleci.com/gh/monocongo/cvdata.svg?style=svg)](https://circleci.com/gh/monocongo/cvdata) -->

# cvdata
Tools for creating and manipulating computer vision datasets

## Installation

This package can be installed into the active Python environment, making the `cvdata` 
module available for import within other Python codes and available for utilization 
at the command line as illustrated in the usage examples below. This package 
is currently supported for Python versions 3.6 and 3.7, and the installation methods below 
assume that the package will be installed into a Python 3.6 or 3.7 virtual environment.

##### From PyPI
This package can be installed into the active Python environment from PyPI via 
`pip`. In addition to installing this package from PyPI, users will also need to 
install the TensorFlow Object Detection API from that project's GitHub repository.
```bash
$ pip install cvdata
$ pip install -e git+https://github.com/tensorflow/models.git#egg=object_detection\&subdirectory=research
```

##### From source
This package can be installed into the active Python environment as source from 
its git repository. We'll first clone/download from GitHub and then install the 
package into the active Python environment:
```bash
$ git clone git@github.com:monocongo/cvdata.git
$ cd cvdata
$ pip install -e .
```

## Resize images
In order to resize images and update the associated annotations use the script 
`cvdata/resize.py` or the corresponding script entry point `cvdata_resize`. This 
script currently supports annotations in KITTI (*.txt) and PASCAL VOC (*.xml) formats. 
For example to resize images to 1024x768 and update the associated annotations in 
KITTI format:
```bash
$ cvdata_resize --input_images /ssd_training/kitti/image_2 \
    --input_annotations /ssd_training/kitti/label_2 \
    --output_images /ssd_training/kitti/image_2 \
    --output_annotations /ssd_training/kitti/label_2 \
    --width 1024 --height 768 --format kitti
```

We can also resize all images in a directory by using the same command as above 
but without an annotation directory or format specified:
```bash
$ cvdata_resize --input_images /ssd_training/kitti/image_2 \
    --output_images /ssd_training/kitti/image_2 \
    --width 1024 --height 768
```

## Rename files
In order to perform bulk renaming of image files we provide the script 
`cvdata/rename` or the corresponding script entry point `cvdata_rename`. This 
allows us to specify a directory containing image files, all of which will be renamed 
according to the `--prefix` (the prefix used for the resulting file names), `--start` 
(the initial number in the enumeration part of the new file names), and `--digits` 
(the width of the enumeration part of the new file names) arguments. For example: 
```bash
$ cvdata_rename --images_dir ~/datasets/handgun/images --prefix handgun --start 100 --digits 6
```
In a future release we'll support renaming of image and corresponding annotation 
files. For example:
```bash
$ cvdata_rename --annotations_dir ~/datasets/handgun/kitti \
>  --images_dir ~/datasets/handgun/images \
> --prefix handgun --start 100 --digits 6 \
> --format kitti --kitti_ids_file file_ids.txt
```

## Annotation format conversion
In order to convert from one annotation format to another use the script 
`cvdata/convert.py` or the corresponding script entry point `cvdata_convert`. This 
script currently supports converting annotations from PASCAL to KITTI, from PASCAL 
to TFRecord, from PASCAL to OpenImages, from KITTI to Darknet, and from KITTI to 
TFRecord. For example: 
```bash
$ cvdata_convert --in_format pascal --out_format kitti \
    --annotations_dir /data/handgun/pascal \
    --images_dir /data/handgun/images \
    --out_dir /data/handgun/kitti \
    --kitti_ids_file handgun.txt

$ cvdata_convert --in_format kitti --out_format tfrecord \
    --annotations_dir /data/kitti \ 
    --images_dir /data/images \
    --out_dir /data/tfrecord/dataset.tfrecord \
    --tf_label_map /data/tfrecord/label_map.pbtxt \
    --tf_shards 2
``` 

## Image format conversion
In order to convert all images in a directory from PNG to JPG we can use the script 
`cvdata/convert.py` or the corresponding script entry point `cvdata_convert`. For 
example:
```bash
$ cvdata_convert --in_format png --out_format jpg --images_dir /datasets/vehicle
```

## Rename annotation labels
In order to rename the image class labels of annotations use the script 
`cvdata/rename.py` or the corresponding script entry point `cvdata_rename`. This 
script currently supports annotations in KITTI (*.txt) and PASCAL VOC (*.xml) 
formats. It is used to replace the label name for all annotation files of the 
specified format in the specified directory. For example:
```bash
$ cvdata_rename.py --labels_dir /data/cvdata/pascal --old handgun --new firearm --format pascal
```

## Exclusion of unwanted images/annotations
Unwanted images and (optionally) their corresponding annotations can be excluded 
(removed) from a dataset using the script `cvdata/exclude.py` or the corresponding 
script entry point `cvdata_exclude`. For example: 
```bash
$ cvdata_exclude --format pascal \
>  --exclusions /data/handgun/exclusions.txt
>  --images /data/handgun/images \
>  --annotations /data/handgun/pascal \
```
The script can also be used to filter out only corresponding image files by omitting 
the `--annotations` argument and corresponding `--format` argument. For example: 
```bash
$ cvdata_exclude --exclusions /data/handgun/exclusions.txt --images /data/handgun/images
```

## Sanitize dataset
In order to clean a dataset's annotations we can utilize the script `cvdata/clean.py` 
or the corresponding script entry point `cvdata_clean` which will convert the images 
to JPG (if any are in PNG format), (optionally) replace labels, (optionally) remove 
bounding boxes that contain specified labels, and update the annotation files so that 
all bounding boxes are within reasonable ranges. If specified then offending/problematic 
files can be moved into a "problems" directory, otherwise they will be removed. 
For example:
```bash
$ cvdata_clean --format pascal \
>    --annotations_dir /data/datasets/delivery_truck/pascal \
>    --images_dir /data/datasets/delivery_truck/images \
>    --problems_dir /data/datasets/delivery_truck/problem \
>    --replace_labels deivery:delivery truck:ups \
>    --remove_labels bus train
```

## Split dataset into training, validation, and test subsets
In order to split a dataset into training, validation, and test subsets we can 
utilize the script `cvdata/split.py` or the corresponding script entry point `cvdata_split`. 
This script's CLI contains options for specifying the source dataset's images and 
annotations directories and the destination images and annotations directories for 
the respective train/valid/test subset splits. The default split ratio is 70% training, 
20% validation, and 10% testing but can be modified with the `--split` argument 
(these are colon-separated float values and should sum to 1). For example: 
```bash
$ cvdata_split --annotations_dir /data/rifle/kitti/label_2 \
> --images_dir /data/rifle/kitti/image_2 \
> --train_annotations_dir /data/rifle/split/kitti/trainval/label_2 \
> --train_images_dir /data/rifle/split/kitti/trainval/image_2 \
> --val_annotations_dir /data/rifle/split/kitti/trainval/label_2 \
> --val_images_dir /data/rifle/split/kitti/trainval/image_2 \
> --test_annotations_dir /data/rifle/split/kitti/test/label_2 \
> --test_images_dir /data/rifle/split/kitti/test/image_2 \
> --format kitti --split 0.65:0.25:0.1 --move
```
In the case where only images are required to be split, we can omit the 
annotations related arguments from the command:
```bash
$ cvdata_split --images_dir /data/rifle/kitti/image_2 \
> --train_images_dir /data/rifle/split/kitti/train/image_2 \
> --val_images_dir /data/rifle/split/kitti/valid/image_2 \
> --test_images_dir /data/rifle/split/kitti/test/image_2 \
> --move
```

## Filtering
The module/script `cvdata/filter.py` or the corresponding script entry point `cvdata_filter` 
can be used to filter the number of image/annotation files of a dataset. It currently 
supports limiting the number of bounding boxes per class type. The filtered dataset 
will contain annotation files with bounding boxes only for the class labels specified 
and limited to the number of boxes specified for each class label. For example: 
```bash
$ cvdata_filter --format darknet \
    --src_annotations /data/darknet \ 
    --dest_annotations /data/filtered_darknet \
    --src_images /data/images \
    --dest_images /data/filtered_images \
    --darknet_labels /data/darknet/labels.txt \
    --boxes_per_class car:6000 truck:6000
```

## Relabel annotations
The module/script `cvdata/relabel.py` or the corresponding script entry point `cvdata_relabel` 
can be used to filter the number of image/annotation files of a dataset. For example, 
to relabel all PASCAL annotation files in a directory from "dog" to "beagle":
```bash
$ cvdata_relabel --labels_dir /data/cvdata/pascal \
  --old dog --new beagle --format pascal
```
Since Darknet (YOLO) annotation files use index values that correspond to entries 
in a class labels file we would use integer values for the `--old` and `--new` 
arguments:
```bash
$ cvdata_relabel --labels_dir /data/cvdata/darknet \
  --old 1 --new 4 --format darknet
```
This function currently supports `darknet`, `kitti`, and `pascal` formats.

## Remove duplicates
The module/script `cvdata/duplicates.py`  or the corresponding script entry point 
`cvdata_duplicates` can be used to remove duplicate images from a directory. This 
works on images that are similar, i.e. images don't need to be exactly the same. 
Optionally the module can remove corresponding annotation files, assuming that the 
annotation file names correspond to the image file names (for example `abc.jpg` and 
`abc.xml`). Also we can move the duplicate files into a separate directory rather 
than removing the files if a directory for duplicates is specified. For example:
```bash
$ cvdata_duplicates --images_dir /data/trucks/ups/images \
>   --annotations_dir /data/trucks/ups/pascal \
>   --dups_dir /data/trucks/ups/dups
```

## Masks
Create masks from region polygons described in an annotation JSON file created by 
the [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/via.html) tool:
```bash
$ cvdata_mask --images /data/images \
>   --annotations /data/via_annotations.json \
>   --masks /data/masks \
>   --format vgg \
>   --classes /data/class_labels.txt
```
Masks will be written with the mask value corresponding to the class ID. For example, 
if we have a class labels file with a single label, then the only class ID is 1 
and so the masks will have a pixel value of (1, 1, 1) where pixels are masked.

By default each mask described in the annotations file will result in a separate 
mask file. So, for example, if the annotation for image file "abc.jpg" includes 
two mask regions then the resulting mask files will be named "abc_0_segmentation.png" 
and "abc_0_segmentation.png". However, if the `--combine` option is used then all 
masks for an images will be included in a single mask file, so the single mask file 
corresponding to image file named "abc.jpg" will be "abc_segmentation.png".

We can also use the `cvdata_mask` script entry point to create TFRecord files 
from an input dataset of JPG images and corresponding PNG masks. For this scenario 
we expect the mask files to have the same base file name as the images files, and 
for the image and mask files to be present in their own separate directories. For 
example:
```bash
$ cvdata_mask --images /data/images --masks /data/masks \
>       --in_format png --out_format tfrecord \
>       --tfrecords /data/tfrecords \
>       --shards 4 -- train_pct 0.8
```
## Visualize annotations
In order to visualize images and corresponding annotations use the script 
`cvdata/visualize.py` or the corresponding script entry point `cvdata_visualize`. 
This script currently supports annotations in COCO (*.json), Darknet (*.txt), KITTI 
(*.txt), TFRecords, and PASCAL VOC (*.xml) formats. It will display bounding boxes 
and labels for all images/annotations in the specified images and annotations 
directories. For example:
```bash
$ cvdata_visualize --format pascal --images_dir /data/weapons/images --annotations_dir /data/weapons/pascal
```

### For developers
##### Testing
Tests are based on `pytest` and are launched in stand-alone virtual environments 
via [tox](https://tox.readthedocs.io/en/latest/): 
```bash
$ tox
```

## Citation
```
@misc {cvdata,
    author = "James Adams",
    title  = "cvdata, an open source Python library for manipulating computer vision datasets",
    url    = "https://github.com/monocongo/cvdata",
    month  = "october",
    year   = "2019--"
}
