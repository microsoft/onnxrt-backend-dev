"""
Run llama model with DORT
========================
"""
import time
import onnxruntime
import numpy as np
import torch
import torch._dynamo.backends.registry
from torch.onnx import ExportOptions
from torch.onnx import _OrtBackend as OrtBackend
from torch.onnx import _OrtBackendOptions as OrtBackendOptions
from onnxrt_backend_dev.llama.llama_helper import get_llama_model
from onnxrt_backend_dev.args import get_parsed_args


args = get_parsed_args(
    "dort_bench",
    description=__doc__,
    backend=("ort", "'ort' or 'inductor'"),
    device=("cpu", "'cpu' or 'cuda'"),
    num_hidden_layers=(1, "number of hidden layers"),
    warmup=5,
    repeat=5,
    expose="backend,repeat,warmup,device,num_hidden_layers",
)

device = "cuda"


def make_aot_ort(dynamic: bool = False):
    ort_session_options = onnxruntime.SessionOptions()
    ort_session_options.log_severity_level = 1
    ort_backend = OrtBackend(
        options=OrtBackendOptions(
            export_options=ExportOptions(
                dynamic_shapes=dynamic,
            ),
            ort_session_options=ort_session_options,
        ),
    )
    return ort_backend, ort_backend


model, example_args_collection = get_llama_model(
    input_dims=[(2, 1024)] * (args.repeat + args.warmup),
    _attn_implementation="eager",
    num_hidden_layers=args.num_hidden_layers,
)


model = model.eval().to(args.device)


local_aot_ort, local_ort = make_aot_ort(dynamic=True)

if args.backend == "ort":
    compiled_model = torch.compile(model, backend=local_ort)
elif args.backend == "inductor":
    compiled_model = torch.compile(model, backend="inductor")
else:
    raise ValueError(f"Unexpected backend={args.backend!r}.")


print("warmup")
start_time = time.perf_counter()
is_cuda = args.device == "cuda"
for i in range(args.warmup):
    result = compiled_model(*example_args_collection[i])
    dummy_loss = torch.ones_like(result[0], memory_format=torch.contiguous_format)
    result[0].backward(dummy_loss)
    if is_cuda:
        torch.cuda.synchronize()
warmup_time = time.perf_counter() - start_time
print(f"warmup done in {warmup_time}s.")

print("measures")
times = []
for example_inputs in example_args_collection[args.warmup :]:
    start_time = time.perf_counter()
    result = compiled_model(*example_inputs)
    dummy_loss = torch.ones_like(result[0], memory_format=torch.contiguous_format)
    result[0].backward(dummy_loss)
    if is_cuda:
        torch.cuda.synchronize()
    times.append(time.perf_counter() - start_time)
print("measures done.")

print(f"backend={args.backend}")
print(f"num_hidden_layers={args.num_hidden_layers}")
print(f"repeat={args.repeat}")
print(f"warmup={args.warmup}")
print(f"device={args.device}")
print(f"avg={np.mean(times)}")
print(f"times={times}")
print(f":time,{np.mean(times)};")
print(f":warmup_time,{warmup_time};")
