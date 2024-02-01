import onnxruntime  # noqa: F401
import os
import unittest
import packaging.version as pv
from onnxrt_backend_dev.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    dump_dort_onnx,
)


def has_cuda():
    import torch

    return torch.cuda.is_available()


def torch_min(v: str) -> bool:
    import torch

    return pv.Version(torch.__version__) < pv.Version(v)


class TestLlamaModel(ExtTestCase):
    @skipif_ci_windows("dynamo compiler is not available on Windows")
    @dump_dort_onnx
    def test_llama_model(self, backend="ort", verbose=__name__ == "__main__"):
        import time
        import torch
        import torch._dynamo.backends.registry
        from torch import optim
        from torch.onnx import ExportOptions
        from torch.onnx import _OrtBackend as OrtBackend
        from torch.onnx import _OrtBackendOptions as OrtBackendOptions
        from transformers import LlamaConfig  # noqa: F811
        from transformers.models.llama.modeling_llama import LlamaModel  # noqa: F811

        config = LlamaConfig(
            num_hidden_layers=1,
            vocab_size=1024,
            hidden_size=32,
            intermediate_size=16,
            max_position_embeddings=1024,  # max_position_embeddings>=vocab_size256 --> introduces graph break
            num_attention_heads=2,
        )
        config._attn_implementation = "eager"
        device = "cuda" if has_cuda() else "cpu"

        def make_aot_ort(dynamic: bool = False):
            ort_session_options = onnxruntime.SessionOptions()
            ort_session_options.log_severity_level = 4
            ort_backend = OrtBackend(
                options=OrtBackendOptions(
                    export_options=ExportOptions(
                        dynamic_shapes=dynamic,
                    ),
                    ort_session_options=ort_session_options,
                ),
            )
            return ort_backend, ort_backend

        class LlamaModelWrapper(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.llama = LlamaModel(config)

            def forward(self, input_ids, attention_mask, position_ids):
                decoder_output = self.llama(
                    input_ids, attention_mask, position_ids, return_dict=False
                )
                return decoder_output[0]

        def generate_example_inputs(batch: int, seq: int):
            # shape: batch x seq x hidden_size
            input_ids = torch.randint(0, 7, size=(batch, seq), dtype=torch.int64).to(
                device
            )
            # Usually, its shape is a tensor with shape batch x seq x seq.
            # However, to bypass some control flow in the model, we use None.
            attention_mask = None
            position_ids = torch.arange(0, seq, dtype=torch.int64).to(device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq).to(device)
            return input_ids, attention_mask, position_ids

        # Reason for using multiple example argument groups:
        #  Export model to ONNX with one example argument group
        #  and test it with other example argument groups.
        example_args_collection = (
            generate_example_inputs(2, 1024),
            generate_example_inputs(2, 1024),
            generate_example_inputs(2, 1024),
            generate_example_inputs(2, 1024),
            generate_example_inputs(2, 1024),
        )

        local_aot_ort, local_ort = make_aot_ort(dynamic=True)
        model = LlamaModelWrapper(config).eval().to(device)
        if backend == "ort":
            compiled_model = torch.compile(model, backend=local_ort)
        elif backend == "inductor":
            compiled_model = torch.compile(model, backend="inductor")

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        if verbose:
            print("warming")
        torch.cuda.synchronize()
        for i, example_inputs in enumerate(example_args_collection):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if verbose:
                    print("--------- forward")
                os.environ["ONNXRT_DUMP_PATH"] += "A"
                result = compiled_model(*example_args_collection[0])
                target = torch.rand_like(result, memory_format=torch.contiguous_format)
                if verbose:
                    print("--------- loss")
                os.environ["ONNXRT_DUMP_PATH"] += "B"
                loss = torch.nn.functional.mse_loss(result, target)
                if verbose:
                    print("--------- backward")
                os.environ["ONNXRT_DUMP_PATH"] += "C"
                loss.backward()
                if verbose:
                    print("--------- step")
                os.environ["ONNXRT_DUMP_PATH"] += "D"
                optimizer.step()
                if verbose:
                    print("--------- zero_grad")
                os.environ["ONNXRT_DUMP_PATH"] += "E"
                optimizer.zero_grad()
                if verbose:
                    print("--------- done")

        torch.cuda.synchronize()

        if verbose:
            print("benchmark")
        os.environ["ONNXRT_DUMP_PATH"] += "Z"
        start_time = time.time()
        for i, example_inputs in enumerate(example_args_collection):
            if verbose:
                print(f"iteration={i}")
            torch.cuda.nvtx.range_push(f"batch{i}")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                torch.cuda.nvtx.range_push("FW")
                result = compiled_model(*example_args_collection[0])
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Loss")
                target = torch.rand_like(result, memory_format=torch.contiguous_format)
                loss = torch.nn.functional.mse_loss(result, target)
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("BW")
                loss.backward()
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Optim")
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Avg time: {(end_time - start_time) / (len(example_args_collection))}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
