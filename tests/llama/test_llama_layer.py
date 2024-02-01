import onnxruntime  # noqa: F401
import copy
import unittest
import packaging.version as pv
from typing import Optional, Tuple
from onnxrt_backend_dev.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
)


def has_cuda():
    import torch

    return torch.cuda.is_available()


def torch_min(v: str) -> bool:
    import torch

    return pv.Version(torch.__version__) < pv.Version(v)


def make_aot_ort(dynamic: bool = False):
    from torch.onnx import (
        _OrtBackend as OrtBackend,
        _OrtBackendOptions as OrtBackendOptions,
        ExportOptions,
    )

    ort_backend = OrtBackend(
        options=OrtBackendOptions(
            export_options=ExportOptions(
                dynamic_shapes=dynamic,
            )
        )
    )
    return ort_backend, ort_backend


class TestLlama(ExtTestCase):
    def _assert_model_numerically(
        self,
        model,
        dynamo_backend,
        example_args_collection_cpu,
        fullgraph: bool = True,
        test_backward: bool = False,
        dynamic: bool = False,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        onnx_export: Optional[str] = None,
        grad_ones=False,
        device=None,
    ):
        import torch

        assert onnx_export, "No export name was given"
        assert device, "No specified device"
        model = model.to(device)
        example_args_collection = [
            tuple(t.to(device) for t in examples)
            for examples in example_args_collection_cpu
        ]

        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend=dynamo_backend,
            dynamic=dynamic,
            fullgraph=fullgraph,
        )
        tested = False

        for example_args in example_args_collection:
            baseline_result = model(*example_args)

            result = compiled_model(*example_args)
            if isinstance(baseline_result, torch.Tensor):
                assert atol > 0
                assert rtol > 0
                torch.testing.assert_close(
                    baseline_result, result, atol=atol, rtol=rtol
                )
                torch.testing.assert_close(
                    baseline_result, result, atol=atol, rtol=rtol
                )
                if test_backward:
                    if grad_ones:
                        dummy_loss = torch.ones_like(
                            baseline_result[0],
                            memory_format=torch.contiguous_format,
                            device=device,
                        )
                        baseline_result[0].backward(dummy_loss)
                        result[0].backward(dummy_loss)
                    else:
                        baseline_result.sum().backward()
                        result.sum().backward()
                    for baseline_param, param in zip(
                        model.parameters(), compiled_model.parameters()
                    ):
                        torch.testing.assert_close(
                            baseline_param.grad, param.grad, atol=atol, rtol=rtol
                        )
                        tested = True
            else:
                if hasattr(baseline_result, "to_tuple"):
                    baseline_result = baseline_result.to_tuple()
                if hasattr(result, "to_tuple"):
                    result = result.to_tuple()
                baseline_result = tuple(b for b in baseline_result if b is not None)
                result = tuple(b for b in result if b is not None)
                assert len(baseline_result) == len(
                    result
                ), f"Mismatch number of outputs {len(baseline_result)} != {len(result)}"
                for baseline_elem, result_elem in zip(baseline_result, result):
                    torch.testing.assert_close(
                        baseline_elem, result_elem, atol=atol, rtol=rtol
                    )
                if test_backward:
                    if grad_ones:
                        dummy_loss = torch.ones_like(
                            baseline_result[0], memory_format=torch.contiguous_format
                        )
                        baseline_result[0].backward(dummy_loss)
                        result[0].backward(dummy_loss)
                    else:
                        torch.testing.assert_close(baseline_result[0], result[0])
                        baseline_result[0].sum().backward()
                        result[0].sum().backward()

                    for baseline_param, param in zip(
                        model.parameters(), compiled_model.parameters()
                    ):
                        torch.testing.assert_close(
                            baseline_param.grad, param.grad, atol=atol, rtol=rtol
                        )
                        tested = True

        if test_backward and not tested:
            raise AssertionError("Model backward was not tested.")

    def _assert_counting_information(
        self,
        ort_backend: "OrtBackend",  # noqa: F821
        expected_execution_count: int,
        number_of_cached_graph_modules: int,
        number_of_exported_onnx_models_for_all_graph_modules: Tuple[int, ...],
        expected_graph_break=0,
    ):
        self.assertEqual(
            expected_execution_count * (expected_graph_break + 1),
            ort_backend.execution_count,
        )
        for (
            onnx_info,
            expected_number_of_onnx_models,
        ) in zip(
            ort_backend._all_ort_execution_info.execution_info_per_graph_module.values(),
            number_of_exported_onnx_models_for_all_graph_modules,
        ):
            self.assertEqual(len(onnx_info), expected_number_of_onnx_models)

    def common_test_model(
        self,
        model,
        example_args_collection,
        test_backward: bool,
        dynamic: bool,
        fullgraph: bool = True,
        onnx_export=None,
        expected_graph_break=0,
        assert_counting=True,
        grad_ones=False,
        device=None,
    ):
        import torch

        torch._dynamo.reset()
        local_aot_ort, local_ort = make_aot_ort(dynamic=dynamic)

        self._assert_model_numerically(
            model,
            local_aot_ort,
            example_args_collection,
            test_backward=test_backward,
            fullgraph=fullgraph,
            onnx_export=onnx_export,
            grad_ones=grad_ones,
            device=device,
        )

        number_of_captured_graphs = 2 if test_backward else 1
        execution_count = len(example_args_collection) * number_of_captured_graphs
        if assert_counting:
            if isinstance(expected_graph_break, int):
                self._assert_counting_information(
                    local_ort,
                    expected_execution_count=execution_count,
                    number_of_cached_graph_modules=number_of_captured_graphs,
                    number_of_exported_onnx_models_for_all_graph_modules=(1,)
                    * number_of_captured_graphs,
                    expected_graph_break=expected_graph_break,
                )
                return
            keep_exc = None
            for value in expected_graph_break:
                try:
                    self._assert_counting_information(
                        local_ort,
                        expected_execution_count=execution_count,
                        number_of_cached_graph_modules=number_of_captured_graphs,
                        number_of_exported_onnx_models_for_all_graph_modules=(1,)
                        * number_of_captured_graphs,
                        expected_graph_break=value,
                    )
                    return
                except AssertionError as e:
                    keep_exc = e
            raise keep_exc

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    def test_ort_mlp_cpu(self):
        import torch

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(2, 4, bias=True)
                self.fc2 = torch.nn.Linear(4, 2, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                return tensor_x

        # with static shape (dynamic=False), the conversion to onnx is done
        # every time the batch size changes
        batch_sizes = [3, 3, 3, 3, 3]

        example_args_collection = tuple(
            (torch.randn(batch, 2, dtype=torch.float32),) for batch in batch_sizes
        )

        self.common_test_model(
            MLP(),
            example_args_collection,
            test_backward=False,
            dynamic=False,
            onnx_export="test_ort_mlp",
            device="cpu",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    def test_ort_mlp_backward_cpu(self):
        import torch

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(2, 4, bias=True)
                self.fc2 = torch.nn.Linear(4, 2, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                return tensor_x

        # with static shape (dynamic=False), the conversion to onnx is done
        # every time the batch size changes
        batch_sizes = [3, 3, 3, 3, 3]

        example_args_collection = tuple(
            (torch.randn(batch, 2, dtype=torch.float32),) for batch in batch_sizes
        )

        self.common_test_model(
            MLP(),
            example_args_collection,
            test_backward=True,
            dynamic=False,
            onnx_export="test_ort_mlp_backward",
            device="cpu",
        )

    @classmethod
    def get_input_dims(cls, dynamic: bool):
        if dynamic:
            input_dims = ((2, 8), (4, 7), (9, 15))
        else:
            input_dims = ((9, 15), (9, 15), (9, 15))
        return input_dims

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    def test_ort_llama_decoder_cpu(self):
        from onnxrt_backend_dev.llama.llama_helper import get_llama_decoder

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_decoder(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=False,
            dynamic=False,
            onnx_export="test_ort_llama_decoder",
            device="cpu",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    def test_ort_llama_decoder_backward_cpu(self):
        from onnxrt_backend_dev.llama.llama_helper import get_llama_decoder

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_decoder(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            onnx_export="test_ort_llama_decoder_backward",
            device="cpu",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    def test_ort_llama_attention_cpu(self):
        from onnxrt_backend_dev.llama.llama_helper import (
            get_llama_attention,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=False,
            dynamic=False,
            onnx_export="test_ort_llama_attention",
            device="cpu",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_ort_llama_attention_cuda(self):
        from onnxrt_backend_dev.llama.llama_helper import (
            get_llama_attention,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=False,
            dynamic=False,
            onnx_export="test_ort_llama_attention",
            device="cuda",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    def test_ort_llama_attention_backward_cpu(self):
        from onnxrt_backend_dev.llama.llama_helper import (
            get_llama_attention,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            onnx_export="test_ort_llama_attention_backward",
            device="cpu",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_ort_llama_attention_backward_cuda(self):
        from onnxrt_backend_dev.llama.llama_helper import (
            get_llama_attention,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            onnx_export="test_ort_llama_attention_backward",
            device="cuda",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    def test_ort_llama_attention_backward_dummy_cpu(self):
        from onnxrt_backend_dev.llama.llama_helper import (
            get_llama_attention,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            onnx_export="test_ort_llama_attention_backward",
            grad_ones=True,
            device="cpu",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    def test_ort_llama_model_cpu(self):
        from onnxrt_backend_dev.llama.llama_helper import (
            get_llama_model,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=False,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_ort_llama_model",
            expected_graph_break=(0, 2),
            device="cpu",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_ort_llama_model_cuda(self):
        from onnxrt_backend_dev.llama.llama_helper import (
            get_llama_model,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=False,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_ort_llama_model",
            expected_graph_break=0,
            device="cuda",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    def test_ort_llama_model_backward_grad_ones_cpu(self):
        from onnxrt_backend_dev.llama.llama_helper import (
            get_llama_model,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_ort_llama_model_backward",
            assert_counting=False,
            grad_ones=True,
            device="cpu",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    def test_ort_llama_model_backward_cpu(self):
        from onnxrt_backend_dev.llama.llama_helper import (
            get_llama_model,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_ort_llama_model_backward",
            assert_counting=False,
            device="cpu",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_ort_llama_model_backward_cuda(self):
        from onnxrt_backend_dev.llama.llama_helper import (
            get_llama_model,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_ort_llama_model_backward",
            assert_counting=False,
            device="cuda",
        )


if __name__ == "__main__":
    # TestLlama().test_ort_llama_model_backward()
    unittest.main(verbosity=2)
