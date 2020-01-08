import os
import setuptools

parent_dir = os.path.dirname(os.path.realpath(__file__))

with open(f"{parent_dir}/README.md", "r") as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name="cvdata",
    version="0.0.3",
    author="James Adams",
    author_email="monocongo@gmail.com",
    description="Tools for creating and manipulating computer vision datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/monocongo/cvdata",
    python_requires="==3.7",
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
        "object_detection @ git+ssh://github.com/tensorflow/models.git@9302933b93f573ac92026ccc48b3b0a4df7b1fda#egg=object_detection&subdirectory=research",
        "opencv-python",
        "pandas",
        "pillow",
        "requests",
        "tensorflow",
        "tqdm",
    ],
)
