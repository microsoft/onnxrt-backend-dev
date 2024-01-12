import unittest
from onnxrt_backend_dev.ext_test_case import ExtTestCase, get_figure
from onnxrt_backend_dev.plotting.data import memory_peak_plot_data
from onnxrt_backend_dev.plotting.memory import memory_peak_plot


class TestPlottingMemory(ExtTestCase):
    def test_memory_peak_plot(self):
        data = memory_peak_plot_data()
        ax = memory_peak_plot(
            data,
            suptitle="nice",
            bars=[55, 110],
            key=("export", "aot", "compute"),
            figsize=(18 * 2, 7 * 2),
        )
        self.assertNotEmpty(ax)
        get_figure(ax).savefig("check.png")


if __name__ == "__main__":
    unittest.main(verbosity=2)
