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

    def _compare(self, a, b, atol, rtol):
        self.assertEqual(type(a), type(b))
        if isinstance(a, tuple):
            self.assertEqual(len(a), len(b))
            for i, j in zip(a, b):
                self._compare(i, j, atol, rtol)
        else:
            import torch

            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)

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

        model2 = copy.deepcopy(model)
        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend=dynamo_backend,
            dynamic=dynamic,
            fullgraph=fullgraph,
        )
        compiled_model2 = torch.compile(
            copy.deepcopy(model2),
            backend=dynamo_backend,
            dynamic=dynamic,
            fullgraph=fullgraph,
        )

        for example_args in example_args_collection:
            not_mixed_result = model2(*example_args)
            not_mixed_loss = not_mixed_result[0].sum()
            not_mixed_loss.backward()

            result2 = compiled_model2(*example_args)
            loss_result2 = result2[0].sum()
            loss_result2.backward()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                baseline_result = model(*example_args)
                result = compiled_model(*example_args)
                loss_baseline = baseline_result[0].sum()
                loss_result = result[0].sum()

            baseline_result = tuple(b for b in baseline_result if b is not None)
            result = tuple(b for b in result if b is not None)
            self._compare(baseline_result, result, atol=atol, rtol=rtol)
            torch.testing.assert_close(loss_baseline, loss_result, atol=atol, rtol=rtol)

            loss_baseline.backward()
            loss_result.backward()
            errors = {}
            gradient = {}
            n_params = 0
            for not_mixed_param, not_mixed_param2, baseline_param, param in zip(
                model2.parameters(),
                compiled_model2.parameters(),
                model.parameters(),
                compiled_model.parameters(),
            ):
                gradient[n_params] = (
                    not_mixed_param.grad,
                    not_mixed_param2.grad,
                    baseline_param.grad,
                    param.grad,
                )
                n_params += 1
                try:
                    torch.testing.assert_close(
                        baseline_param.grad, param.grad, atol=atol, rtol=rtol
                    )
                except Exception as e:
                    errors[n_params - 1] = (
                        e,
                        baseline_param.grad,
                        param.grad,
                        baseline_param.name,
                    )
            if errors:
                rows = ["not mixed=A, mixed=X, TORCH=T, DORT=D"]
                for k, v in gradient.items():
                    d1 = torch.abs(v[0] - v[2]).max()
                    d2 = torch.abs(v[0] - v[1]).max()
                    d3 = torch.abs(v[2] - v[3]).max()
                    minis = " ".join([f"{_.min():1.2f}" for _ in v])
                    maxis = " ".join([f"{_.max():1.2f}" for _ in v])
                    rows.append(
                        f"{k:02d}: TA/TM={d1:1.4f} TA/DA={d2:1.4f} TX/DX={d3:1.4f} shape: {v[2].shape} [{minis}-{maxis}]"
                    )
                for k, v in errors.items():
                    diff = torch.abs(v[2] - v[1]).max()
                    errs = str(v[0]).replace("\n", " --- ")
                    rows.append(f"{k:02d}: name={v[3]!r} abs_err={diff} err={errs}")
                msg = "\n".join(rows)
                raise AssertionError(f"{n_params} gradient\n{msg}")

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
        atol=1e-4,
        rtol=1e-4,
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
            atol=atol,
            rtol=rtol,
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
            fullgraph=True,
            onnx_export="test_ort_llama_mixed",
            expected_graph_break=7,
            assert_counting=False,
            device="cuda",
            rtol=5e-2,
        )


if __name__ == "__main__":
    # TestLlama().test_ort_llama_model_backward_nofullgraph()
    unittest.main(verbosity=2)
