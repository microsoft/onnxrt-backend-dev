"""
Measure LLAMA speed
===================
"""

import pandas
import matplotlib.pyplot as plt
from onnxrt_backend_dev.bench_run import run_benchmark

repeat = 5
script_name = "onnxrt_backend_dev.llama.dort_bench"

configs = [
    dict(backend="ort", device="cpu", num_hidden_layers=1, repeat=repeat),
    dict(backend="ort", device="cpu", num_hidden_layers=2, repeat=repeat),
]


data = run_benchmark(script_name, configs, verbose=1)
df = pandas.DataFrame(data)
df = df.drop(["ERROR", "OUTPUT"], axis=1)
filename = "plot_llama_bench.csv"
df.to_csv(filename, index=False)
df = pandas.read_csv(filename)  # to cast type
print(df)

################################
# More simple

dfs = df[["num_hidden_layers", "time", "warmup_time"]]
print(dfs)

###############################
# Plot.

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
dfs[["num_hidden_layers", "time"]].set_index("num_hidden_layers").plot(
    title=f"llama with dort\ntime for {repeat} iterations", ax=ax[0]
)
dfs[["num_hidden_layers", "warmup_time"]].set_index("num_hidden_layers").plot(
    title="llama with dort\nwarmup time", ax=ax[1]
)
fig.tight_layout()
fig.savefig("plot_llama_bench.png")
