import multiprocessing
import platform
import re
import subprocess
import sys
from typing import Dict, List, Union


def get_machine() -> Dict[str, Union[str, int, float]]:
    """
    Returns the machine specification.
    """
    cpu = dict(
        machine=str(platform.machine()),
        processor=str(platform.processor()),
        version=str(sys.version),
        cpu=int(multiprocessing.cpu_count()),
        executable=str(sys.executable),
    )
    try:
        import torch.cuda
    except ImportError:
        return cpu

    cpu["has_cuda"] = torch.cuda.is_available()
    if cpu["has_cuda"]:
        cpu["capability"] = torch.cuda.get_device_capability(0)
        cpu["device_name"] = torch.cuda.get_device_name(0)
    return cpu


def _cmd_line(
    script_name: str, **kwargs: Dict[str, Union[str, int, float]]
) -> List[str]:
    args = [sys.executable, "-m", script_name]
    for k, v in kwargs.items():
        args.append(f"--{k}")
        args.append(str(v))
    return args


def _extract_metrics(text: str) -> Dict[str, str]:
    reg = re.compile(":(.*?),(.*.?);")
    res = reg.findall(text)
    if len(res) == 0:
        return {}
    return dict(res)


def run_benchmark(
    script_name: str, configs: List[Dict[str, Union[str, int, float]]], verbose: int = 0
) -> List[Dict[str, str]]:
    """
    Runs a script multiple times and extract information from the output
    following the pattern ``:<metric>,<value>;``.

    :param script_name: python script to run
    :param configs: list of execution to do
    :param verbose: use tqdm to follow the progress
    :return: values
    """
    if verbose:
        from tqdm import tqdm

        loop = tqdm(configs)
    else:
        loop = configs

    machine = get_machine()
    data = []
    for config in loop:
        cmd = _cmd_line(script_name, **config)

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res = p.communicate()
        out, err = res

        sout = out.decode("utf-8", errors="ignore")
        serr = err.decode("utf-8", errors="ignore")
        metrics = _extract_metrics(sout)
        metrics.update(config)
        metrics["ERROR"] = serr
        metrics["OUTPUT"] = sout
        metrics.update(machine)
        metrics["CMD"] = f"[{' '.join(cmd)}]"
        data.append(metrics)
    return data
