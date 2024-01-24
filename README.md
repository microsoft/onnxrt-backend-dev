
[![PyPi version](https://badge.fury.io/py/onnxrt-backend-dev.svg)](http://badge.fury.io/py/onnxrt-backend-dev)
[![GitHub Issues](http://img.shields.io/github/issues/microsoft/onnxrt-backend-dev.png)](https://github.com/microsoft/onnxrt-backend-dev/issues)
[![Black](https://img.shields.io/github/repo-size/microsoft/onnxrt-backend-dev)](https://github.com/psf/black)
[![Code Coverage](https://codecov.io/gh/microsoft/onnxrt-backend-dev/branch/main/graph/badge.svg)](https://codecov.io/gh/microsoft/onnxrt-backend-dev)

# onnxrt-backend-dev: tools to help developping onnx functionalities in torch

[onnxrt-backend-dev documentation](https://microsoft.github.io/onnxrt-backend-dev/)

## Getting started

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

## Highlights

### Compare torch exporters

The script evaluates the memory peak, the computation time of the exporters.
It also compares the exported models when run through onnxruntime.
The full script takes around 20 minutes to complete. It stores on disk
all the graphs, the data used to draw them, and the models.

::

    python docs/examples/plot_torch_export.py -s large

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

