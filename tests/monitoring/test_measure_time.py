import math
import unittest
from onnxrt_backend_dev.ext_test_case import ExtTestCase
from onnxrt_backend_dev.monitoring.benchmark import measure_time


class TestMeasureTime(ExtTestCase):
    def test_measure_time(self):
        res = measure_time(lambda: math.cos(0.5))
        self.assertIsInstance(res, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
