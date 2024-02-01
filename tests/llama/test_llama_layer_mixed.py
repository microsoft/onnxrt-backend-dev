import onnxruntime  # noqa: F401
import copy
import pprint
import unittest
import packaging.version as pv
from typing import Optional, Tuple
from onnxrt_backend_dev.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
    dump_dort_onnx,
)


def cuda_available():
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


def _pprint_output(obj):
    if isinstance(obj, (list, tuple)):
        return [_pprint_output(o) for o in obj]
    if hasattr(obj, "shape"):
        return f"T[{obj.shape}:{obj.dtype}]"
    raise AssertionError(f"Unable to pretty print type {type(obj)}.")


def pprint_output(obj):
    return pprint.pformat(_pprint_output(obj))


class TestLlamaMixed(ExtTestCase):
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

        one_example = None
        for example_args in example_args_collection:
            one_example = example_args

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                baseline_result = model(*example_args)
                result = compiled_model(*example_args)
                loss_baseline = baseline_result[0].sum()
                loss_result = result[0].sum()

            loss_baseline.backward()
            loss_result.backward()
            for baseline_param, param in zip(
                model.parameters(), compiled_model.parameters()
            ):
                torch.testing.assert_close(
                    baseline_param.grad, param.grad, atol=atol, rtol=rtol
                )

        # export to onnx
        try:
            torch.onnx.export(
                copy.deepcopy(model), *one_example, f"{onnx_export}_script.onnx"
            )
        except Exception as e:
            print("torch.onnx.export failed:", e)
        try:
            torch.onnx.dynamo_export(copy.deepcopy(model), *one_example).save(
                f"{onnx_export}_dynamo.onnx"
            )
        except Exception as e:
            print("torch.onnx.dynamo_export failed:", e)

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
        self.assertEqual(
            len(ort_backend._all_ort_execution_info.execution_info_per_graph_module),
            number_of_cached_graph_modules * (expected_graph_break + 1),
        )
        self.assertEqual(
            len(ort_backend._all_ort_execution_info.execution_info_per_graph_module),
            len(number_of_exported_onnx_models_for_all_graph_modules)
            * (expected_graph_break + 1),
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
            device=device,
        )

        number_of_captured_graphs = 2 if test_backward else 1
        execution_count = len(example_args_collection) * number_of_captured_graphs
        if assert_counting:
            self._assert_counting_information(
                local_ort,
                expected_execution_count=execution_count,
                number_of_cached_graph_modules=number_of_captured_graphs,
                number_of_exported_onnx_models_for_all_graph_modules=(1,)
                * number_of_captured_graphs,
                expected_graph_break=expected_graph_break,
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
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    @unittest.skipIf(not cuda_available(), reason="always works on cuda")
    @dump_dort_onnx
    def test_ort_llama_mixed_cuda(self):
        from onnxrt_backend_dev.llama.llama_helper import (
            get_llama_model,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(
            input_dims=input_dims,
            device="cuda",
            _attn_implementation="eager",
        )
        self.common_test_model(
            model,
            example_args_collection,
            True,
            False,
            fullgraph=False,
            onnx_export="test_ort_llama_mixed",
            expected_graph_break=7,
            assert_counting=False,
            device="cuda",
        )


if __name__ == "__main__":
    # TestLlama().test_ort_llama_model_backward_nofullgraph()
    unittest.main(verbosity=2)
