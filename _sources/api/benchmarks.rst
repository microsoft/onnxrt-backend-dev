
==========
benchmarks
==========

onnxrt_backend_dev.bench_run
============================

BenchmarkError
++++++++++++++

.. autoclass:: onnxrt_backend_dev.bench_run.BenchmarkError

get_machine
+++++++++++

.. autofunction:: onnxrt_backend_dev.bench_run.get_machine

run_benchmark
+++++++++++++

.. autofunction:: onnxrt_backend_dev.bench_run.run_benchmark

scripts
=======

onnxrt_backend_dev.llama.dort_bench
+++++++++++++++++++++++++++++++++++

The script runs a few iterations of a dummy llama model.
See :ref:`l-plot-llama-bench`. The script can be called multiple times by 
function :func:`onnxrt_backend_dev.bench_run.run_benchmark` to collect many figures.

::

    python -m onnxrt_backend_dev.llama.dort_bench --help

::

    options:
        -h, --help            show this help message and exit
        -r REPEAT, --repeat REPEAT
                                number of times to repeat the measure, default is 5
        -w WARMUP, --warmup WARMUP
                                number of iterations to warmup, includes the settings, default is 5
        --backend BACKEND     'ort' or 'inductor', default is ort
        --device DEVICE       'cpu' or 'cuda', default is cpu
        --num_hidden_layers NUM_HIDDEN_LAYERS
                                number of hidden layers, default is 1