# cvdata
Tools for creating and manipulating computer vision datasets

## OpenImages
To download various image classes from [OpenImages](https://storage.googleapis.com/openimages/web/index.html) 
use the script `cvdata/openimages.py`. This script currently only supports writing annotations in PASCAL VOC format.
For example:
```bash
$ python cvdata/openimages.py --label Handgun Shotgun Rifle --exclusions /home/james/git/cvdata/exclusions/exclusions_weapons.txt --base_dir /data/cvdata/weapons --format pascal
```
The above will save each image class in a separate subdirectory under the base 
directory, with images in a subdirectory named "images" and the PASCAL VOC format 
annotations in a subdirectory named "pascal".

## Visualize annotations
In order to visualize images and corresponding annotations use the script 
`cvdata/visualize.py`. This script currently supports annotations in COCO (*.json), 
Darknet (*.txt), KITTI (*.txt), and PASCAL VOC (*.xml) formats. It will display 
bounding boxes and labels for all images/annotations in the specified images and 
annotations directories. For example:
```bash
$ python cvdata/visualize.py --format pascal --images_dir /data/weapons/images --annotations_dir /data/weapons/pascal
```
