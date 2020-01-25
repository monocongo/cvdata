import os
import setuptools

parent_dir = os.path.dirname(os.path.realpath(__file__))

with open(f"{parent_dir}/README.md", "r") as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name="cvdata",
    version="0.0.6",
    author="James Adams",
    author_email="monocongo@gmail.com",
    description="Tools for creating and manipulating computer vision datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/monocongo/cvdata",
    python_requires="==3.7.*",
    packages=[
        "cvdata",
    ],
    provides=[
        "cvdata",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "boto3",
        "contextlib2",
        "lxml",
        "ImageHash",
        "opencv-python",
        "pandas",
        "pillow",
        "requests",
        "tensorflow",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "cvdata_analyze=cvdata.analyze:main",
            "cvdata_clean=cvdata.clean:main",
            "cvdata_convert=cvdata.convert:main",
            "cvdata_duplicates=cvdata.duplicates:main",
            "cvdata_exclude=cvdata.exclude:main",
            "cvdata_filter=cvdata.filter:main",
            "cvdata_openimages=cvdata.openimages:main",
            "cvdata_mask=cvdata.mask:main",
            "cvdata_relabel=cvdata.relabel:main",
            "cvdata_rename=cvdata.rename:main",
            "cvdata_resize=cvdata.resize:main",
            "cvdata_split=cvdata.split:main",
            "cvdata_visualize=cvdata.visualize:main",
        ]
    },
)
