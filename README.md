# cvdata
Tools for creating and manipulating computer vision datasets

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

## Rename annotation labels
In order to rename the image class labels of annotations use the script 
`cvdata/rename.py`. This script currently supports annotations in KITTI (*.txt) 
and PASCAL VOC (*.xml) formats. It is used to replace the label name for all 
annotation files of the specified format in the specified directory. For example:
```bash
$ python rename.py --labels_dir /data/cvdata/pascal --old handgun --new firearm --format pascal
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
