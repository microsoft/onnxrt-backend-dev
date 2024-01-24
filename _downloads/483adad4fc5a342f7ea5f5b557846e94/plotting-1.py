import matplotlib.pyplot as plt
from onnxrt_backend_dev.plotting.data import memory_peak_plot_data
from onnxrt_backend_dev.plotting.memory import memory_peak_plot

data = memory_peak_plot_data()
ax = memory_peak_plot(
    data,
    suptitle="nice",
    bars=[55, 110],
    key=("export", "aot", "compute"),
    figsize=(18 * 2, 7 * 2),
)
plt.show()