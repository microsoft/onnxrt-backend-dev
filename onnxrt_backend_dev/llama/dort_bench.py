"""
Run llama model with DORT
=========================

The script runs a few iterations of a dummy llama model.

::

    python -m onnxrt_backend_dev.llama.dort_bench --help
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
    backend=("ort", "'ort' or 'inductor' or 'eager'"),
    device=("cpu", "'cpu' or 'cuda'"),
    num_hidden_layers=(1, "number of hidden layers"),
    warmup=5,
    repeat=5,
    mixed=(0, "mixed precision (based on autocast)"),
    export=(
        "",
        "export the model with dynamo and torch.script, " "use this as a prefix",
    ),
    expose="backend,repeat,warmup,device,num_hidden_layers,mixed,export",
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
elif args.backend == "eager":
    compiled_model = model
else:
    raise ValueError(f"Unexpected backend={args.backend!r}.")


def loop_iteration(is_cuda, inputs, compiled_model):
    if args.mixed and is_cuda:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = compiled_model(*inputs)
    else:
        result = compiled_model(*inputs)

    dummy_loss = torch.ones_like(result[0], memory_format=torch.contiguous_format)
    result[0].backward(dummy_loss)
    if is_cuda:
        torch.cuda.synchronize()


if args.export:
    from onnxrewriter.optimizer import optimize

    providers = (
        ["CPUExecutionProvider"]
        if device == "cpu"
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    filename = f"{args.export}.script.onnx"
    print("export with torch.onnx.export to {filename!r}")
    input_names = ["input{i}" for i in len(example_args_collection[0])]
    torch.onnx.export(model, *example_args_collection[0], filename, input_names)

    ofilename = f"{args.export}.script.opt.onnx"
    print("onnxruntime optimization to {ofilename!r}")
    opts = onnxruntime.SessionOptions()
    opts.optimized_model_filepath = ofilename
    sess = onnxruntime.InferenceSession(filename, opts, providers=providers)

    filename = f"{args.export}.dynamo.onnx"
    print("export with torch.onnx.dynamo_export to {filename!r}")
    export_output = torch.onnx.dynamo_export(model, *args)
    optimized_model = optimize(export_output.model_proto)
    with open(filename, "wb") as f:
        f.write(optimized_model.SerializeToString())

    ofilename = f"{args.export}.dynamo.opt.onnx"
    print("onnxruntime optimization to {ofilename!r}")
    opts = onnxruntime.SessionOptions()
    opts.optimized_model_filepath = ofilename
    sess = onnxruntime.InferenceSession(filename, opts, providers=providers)

print("warmup")
warmup_times = []
is_cuda = args.device == "cuda"
for i in range(args.warmup):
    example_inputs = example_args_collection[i]
    inputs = [t.to("cuda") for t in example_inputs] if is_cuda else example_inputs
    start_time = time.perf_counter()
    loop_iteration(is_cuda, inputs, compiled_model)
    warmup_times.append(time.perf_counter() - start_time)

warmup_time = time.perf_counter() - start_time
print(f"warmup done in {warmup_time}s.")

print("measures")
times = []
for example_inputs in example_args_collection[args.warmup :]:
    inputs = [t.to("cuda") for t in example_inputs] if is_cuda else example_inputs
    start_time = time.perf_counter()
    loop_iteration(is_cuda, inputs, compiled_model)
    times.append(time.perf_counter() - start_time)

print("measures done.")

print(f"backend={args.backend}")
print(f"num_hidden_layers={args.num_hidden_layers}")
print(f"mixed={args.mixed}")
print(f"repeat={args.repeat}")
print(f"device={args.device}")
print(f"avg={np.mean(times)}")
print(f"times={times}")
print(f"warmup_times={warmup_times}")
print(f":time,{np.mean(times)};")
print(f":warmup_time,{sum(warmup_times)};")
print(f":torch,{torch.__file__};")
