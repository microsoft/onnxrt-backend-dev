"""
.. _l-plot-llama-bench:

Measure LLAMA speed
===================
"""

import pandas
import matplotlib.pyplot as plt
import itertools
import torch
from onnxrt_backend_dev.ext_test_case import unit_test_going
from onnxrt_backend_dev.bench_run import run_benchmark, get_machine, BenchmarkError

repeat = 5
script_name = "onnxrt_backend_dev.llama.dort_bench"
machine = {} if unit_test_going() else get_machine()

if machine.get("capability", (0, 0)) >= (7, 0):
    configs = []
    for backend, device, num_hidden_layers, mixed in itertools.product(
        ["eager", "inductor", "ort"],
        ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
        [1, 2],
        [0, 1],
    ):
        if mixed == 1 and device == "cpu":
            continue
        configs.append(
            dict(
                backend=backend,
                device=device,
                num_hidden_layers=num_hidden_layers,
                repeat=repeat,
                mixed=mixed,
            )
        )
else:
    configs = [
        dict(backend="ort", device="cpu", num_hidden_layers=1, repeat=repeat, mixed=0),
        dict(backend="ort", device="cpu", num_hidden_layers=2, repeat=repeat, mixed=0),
    ]


try:
    data = run_benchmark(script_name, configs, verbose=1)
    data_collected = True
except BenchmarkError as e:
    print(e)
    data_collected = False

if data_collected:
    df = pandas.DataFrame(data)
    df = df.drop(["ERROR", "OUTPUT"], axis=1)
    filename = "plot_llama_bench.csv"
    df.to_csv(filename, index=False)
    df = pandas.read_csv(filename)  # to cast type
    print(df)

################################
# More simple

columns = ["backend", "num_hidden_layers", "mixed", "time", "device", "warmup_time"]
if data_collected:
    try:
        dfs = df[columns]
    except KeyError as e:
        raise RuntimeError(f"Missing columns in {df.columns}\n{df.head().T}") from e
    print(dfs)

###############################
# Plot.

if data_collected:
    fig, ax = plt.subplots(2, 3, figsize=(10, 9))

    # warmup time

    piv = dfs[(dfs.device == "cpu") & (dfs.mixed == 0)].pivot(
        index="num_hidden_layers", columns="backend", values="warmup_time"
    )
    if len(piv) > 0:
        piv.plot(title="llama with dort on cpu\nwarmup time", ax=ax[0, 0])

    piv = dfs[(dfs.device == "cuda") & (dfs.mixed == 0)].pivot(
        index="num_hidden_layers", columns="backend", values="warmup_time"
    )
    if len(piv) > 0:
        piv.plot(title="llama with dort on cuda\nwarmup time", ax=ax[0, 1])

    piv = dfs[(dfs.device == "cuda") & (dfs.mixed == 0)].pivot(
        index="num_hidden_layers", columns="backend", values="warmup_time"
    )
    if len(piv) > 0:
        piv.plot(title="llama with dort on cuda (mixed)\nwarmup time", ax=ax[0, 2])

    # time

    piv = dfs[(dfs.device == "cpu") & (dfs.mixed == 0)].pivot(
        index="num_hidden_layers", columns="backend", values="time"
    )
    if len(piv) > 0:
        piv.plot(
            title=f"llama with dort on cpu\ntraining time for {repeat} iterations",
            ax=ax[1, 0],
        )

    piv = dfs[(dfs.device == "cuda") & (dfs.mixed == 0)].pivot(
        index="num_hidden_layers", columns="backend", values="time"
    )
    if len(piv) > 0:
        piv.plot(
            title=f"llama with dort on cuda\ntraining time for {repeat} iterations",
            ax=ax[1, 1],
        )

    piv = dfs[(dfs.device == "cuda") & (dfs.mixed == 1)].pivot(
        index="num_hidden_layers", columns="backend", values="time"
    )
    if len(piv) > 0:
        piv.plot(
            title=f"llama with dort on cuda (mixed)\ntraining time for {repeat} iterations",
            ax=ax[1, 2],
        )

    fig.tight_layout()
    fig.savefig("plot_llama_bench.png")
