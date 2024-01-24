
========
Tutorial
========

Getting Started
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
