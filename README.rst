
.. image:: https://github.com/microsoft/onnxrt-backend-dev/raw/main/docs/_static/logo.png
    :width: 120

===========================================================================
onnxrt-backend-dev: tools to help developping onnx functionalities in torch
===========================================================================


.. image:: https://badge.fury.io/py/onnxrt-backend-dev.svg
    :target: http://badge.fury.io/py/onnxrt-backend-dev

.. image:: http://img.shields.io/github/issues/microsoft/onnxrt-backend-dev.png
    :alt: GitHub Issues
    :target: https://github.com/microsoft/onnxrt-backend-dev/issues

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: https://opensource.org/license/MIT/

.. image:: https://img.shields.io/github/repo-size/microsoft/onnxrt-backend-dev
    :target: https://github.com/microsoft/onnxrt-backend-dev/
    :alt: size

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://codecov.io/gh/microsoft/onnxrt-backend-dev/branch/main/graph/badge.svg?token=Wb9ZGDta8J 
    :target: https://codecov.io/gh/microsoft/onnxrt-backend-dev

Getting started
===============

pytorch nightly build should be installed, see
`Start Locally <https://pytorch.org/get-started/locally/>`_.
The following instructions may need to be updated based on your configurtion
(cuda version, os).

::

    git clone https://github.com/microsoft/onnxrt-backend-dev.git
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
    pip install onnxruntime-training pynvml
    pip install -r requirements-dev.txt    
    export PYTHONPATH=$PYTHONPATH:<this folder>

Then install *onnx-script* and *onnx-rewriter*.

Interesting examples
====================

Compare torch exporters
+++++++++++++++++++++++

The script evaluates the memory peak, the computation time of the exporters.
It also compares the exported models when run through onnxruntime.
The full script takes around 20 minutes to complete. It stores on disk
all the graphs, the data used to draw them, and the models.

::

    python docs/examples/plot_torch_export.py -s large
