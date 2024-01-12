# -*- coding: utf-8 -*-
import os

from setuptools import setup

######################
# beginning of setup
######################


here = os.path.dirname(__file__)
if here == "":
    here = "."
package_data = {}

try:
    with open(os.path.join(here, "requirements.txt"), "r") as f:
        requirements = f.read().strip(" \n\r\t").split("\n")
except FileNotFoundError:
    requirements = []
if not requirements or requirements == [""]:
    requirements = ["numpy", "onnx"]

try:
    with open(os.path.join(here, "README.rst"), "r", encoding="utf-8") as f:
        long_description = (
            "onnxrt-backend-dev:" + f.read().split("onnxrt-backend-dev:")[1]
        )
except FileNotFoundError:
    long_description = ""

version_str = "0.1.0"
with open(os.path.join(here, "onnxrt_backend_dev/__init__.py"), "r") as f:
    line = [
        _
        for _ in [_.strip("\r\n ") for _ in f.readlines()]
        if _.startswith("__version__")
    ]
    if line:
        version_str = line[0].split("=")[1].strip('" ')


setup(
    name="onnxrt-backend-dev",
    version=version_str,
    description="Developping tools for torch/onnx",
    long_description=long_description,
    author="Microsoft Corporation",
    author_email="onnx@microsoft.com",
    url="https://github.com/microsoft/onnxrt-backend-dev",
    package_data=package_data,
    setup_requires=["numpy", "onnx"],
    install_requires=requirements,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
