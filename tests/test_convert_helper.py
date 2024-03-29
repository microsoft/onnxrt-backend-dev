import onnxruntime  # noqa: F401
import unittest
from onnx import ModelProto
from onnxrt_backend_dev.ext_test_case import ExtTestCase
from onnxrt_backend_dev.convert_helper import (
    optimize_model_proto,
    inline_model_proto,
    ort_optimize,
)


def has_cuda():
    import torch

    return torch.cuda.is_available()


def has_onnxrewriter():
    try:
        import onnxrewriter  # noqa: F401

        return True
    except ImportError:
        return False


input_dims = ((2, 1024),)


class TestConvertHelper(ExtTestCase):
    @unittest.skipIf(not has_onnxrewriter(), "onnx-rewriter not installed")
    def test_optimize_llama(self):
        import torch
        from onnxrt_backend_dev.llama.llama_helper import get_llama_attention

        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        model(*example_args_collection[0])
        model = torch.onnx.dynamo_export(model, *example_args_collection[0])
        model_proto = model.model_proto
        model_proto = optimize_model_proto(model_proto)
        self.assertIsInstance(model_proto, ModelProto)

    def test_inline_llama(self):
        import torch
        from onnxrt_backend_dev.llama.llama_helper import get_llama_attention

        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        model(*example_args_collection[0])
        model = torch.onnx.dynamo_export(model, *example_args_collection[0])
        model_proto = model.model_proto
        model_proto = inline_model_proto(model_proto)
        self.assertIsInstance(model_proto, ModelProto)

    def test_ort_optimize(self):
        import torch
        from onnxrt_backend_dev.llama.llama_helper import get_llama_attention

        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        model(*example_args_collection[0])
        model = torch.onnx.dynamo_export(model, *example_args_collection[0])
        model_proto = model.model_proto
        ort_optimize(model_proto, providers="cpu", output="test_ort_optimize.onnx")

    @unittest.skipIf(not has_cuda(), reason="no cuda")
    def test_ort_optimize_cuda(self):
        import torch
        from onnxrt_backend_dev.llama.llama_helper import get_llama_attention

        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        model(*example_args_collection[0])
        model = torch.onnx.dynamo_export(model, *example_args_collection[0])
        model_proto = model.model_proto
        ort_optimize(
            model_proto, providers="cuda", output="test_ort_optimize_cuda.onnx"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
