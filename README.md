# cvdata
Tools for creating and manipulating computer vision datasets

## Installation
This package can be installed into the active Python environment via `pip`, making 
the `cvdata` module available for import within other Python code:
```bash
$ pip install cvdata
```
To utilize the package from command line as illustrated in the usage examples below 
please clone/download from GitHub:
```bash
$ git clone git@github.com:monocongo/cvdata.git
```

## OpenImages
To download various image classes from [OpenImages](https://storage.googleapis.com/openimages/web/index.html) 
use the script `cvdata/openimages.py`. This script currently only supports writing 
annotations in PASCAL VOC format. For example:
```bash
$ python cvdata/openimages.py --label Handgun Shotgun Rifle --exclusions /home/james/git/cvdata/exclusions/exclusions_weapons.txt --base_dir /data/cvdata/weapons --format pascal
```
The above will save each image class in a separate subdirectory under the base 
directory, with images in a subdirectory named "images" and the PASCAL VOC format 
annotations in a subdirectory named "pascal".

## Resize images
In order to resize images and update the associated annotations use the script 
`cvdata/resize.py`. This script currently supports annotations in KITTI (*.txt) 
and PASCAL VOC (*.xml) formats. For example to resize images to 1024x768 and 
update the associated annotations in KITTI format:
```bash
$ python resize.py --input_images /ssd_training/kitti/image_2 \
    --input_annotations /ssd_training/kitti/label_2 \
    --output_images /ssd_training/kitti/image_2 \
    --output_annotations /ssd_training/kitti/label_2 \
    --width 1024 --height 768 --format kitti
```

We can also resize all images in a directory by using the same command as above 
but without an annotation directory, extension, or format specified:
```bash
$ python resize.py --input_images /ssd_training/kitti/image_2 \
    --output_images /ssd_training/kitti/image_2 \
    --width 1024 --height 768
```

## Rename files
In order to perform bulk renaming of image files we provide the script 
`cvdata/rename.py`. This allows us to specify a directory containing image files, 
all of which will be renamed according to the `--prefix` (the prefix used for the 
resulting file names), `--start` (the initial number in the enumeration part of 
the new file names), and `--digits` (width of the enumeration part of the new 
file names) arguments. For example: 
```bash
$ python rename.py --images_dir ~/datasets/handgun/images --prefix handgun --start 100 --digits 6
```
In a future release we'll support renaming of image and corresponding annotation 
files. For example:
```bash
$ python rename.py --annotations_dir ~/datasets/handgun/kitti \
>  --images_dir ~/datasets/handgun/images \
> --prefix handgun --start 100 --digits 6 \
> --format kitti --kitti_ids_file file_ids.txt
```

## Convert annotation formats
In order to convert from one annotation format to another use the script 
`cvdata/convert.py`. This script currently supports converting annotations from 
PASCAL to KITTI formats. For example: 
```bash
$ python convert.py --in_format pascal --out_format kitti \
    --annotations_dir /data/handgun/pascal \
    --images_dir /data/handgun/images \
    --out_dir /data/handgun/kitti \
    --kitti_ids_file handgun.txt
``` 

## Image format conversion
In order to convert all images in a directory from PNG to JPG we can use the script 
`cvdata/convert.py`. For example:
```bash
$ python convert.py --in_format png --out_format jpg --images_dir /datasets/vehicle
```

## Rename annotation labels
In order to rename the image class labels of annotations use the script 
`cvdata/rename.py`. This script currently supports annotations in KITTI (*.txt) 
and PASCAL VOC (*.xml) formats. It is used to replace the label name for all 
annotation files of the specified format in the specified directory. For example:
```bash
$ python rename.py --labels_dir /data/cvdata/pascal --old handgun --new firearm --format pascal
```

## Sanitize dataset
In order to clean a dataset's annotations we can utilize the script `cvdata/clean.py` 
which will convert the images to JPG (if any are in PNG format), rename labels 
(if specified) and update the PASCAL VOC annotation files so that all bounding 
boxes are within reasonable range. For example:
```bash
$ python clean.py --format pascal \
>     --annotations_dir /data/datasets/delivery_truck/pascal \
>    --images_dir /data/datasets/delivery_truck/images \
>    --rename_labels deivery:delivery
```

## Split dataset into training, validation, and test subsets
In order to split a dataset into training, validation, and test subsets we can 
utilize the script `cvdata/split.py`. This script's CLI contains options for 
specifying the source dataset's images and annotations directories and the destination 
images and annotations directories for the respective train/valid/test subset splits. 
The default split ratio is 70% training, 20% validation, and 10% testing but can 
be modified with the `--split` argument (these are colon-separated float 
values and should sum to 1). For example: 
```bash
$ python split.py --annotations_dir /data/rifle/kitti/label_2 \
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
annotations-related arguments from the command:
```bash
$ python split.py --images_dir /data/rifle/kitti/image_2 \
> --train_images_dir /data/rifle/split/kitti/train/image_2 \
> --val_images_dir /data/rifle/split/kitti/valid/image_2 \
> --test_images_dir /data/rifle/split/kitti/test/image_2 \
> --move
```

## Visualize annotations
In order to visualize images and corresponding annotations use the script 
`cvdata/visualize.py`. This script currently supports annotations in COCO (*.json), 
Darknet (*.txt), KITTI (*.txt), and PASCAL VOC (*.xml) formats. It will display 
bounding boxes and labels for all images/annotations in the specified images and 
annotations directories. For example:
```bash
$ python cvdata/visualize.py --format pascal --images_dir /data/weapons/images --annotations_dir /data/weapons/pascal
```
