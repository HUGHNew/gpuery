"""
parse the default environment variables: CUDA_VISIBLE_DEVICES

If CUDA_VISIBLE_DEVICES is not set, output warning message and use the idle GPU with minimal index.

customized environment variables: (unit: percentage)
- POWER_THRES
- UTIL_THRES
- TMEP_THRES
- MEM_THRES
"""

import os, warnings
from typing import List
from functools import partial

from .entity import GpuStatus, BaseRatio, Power, Memory


def parse_num(i: str, parse_fn, deft):
    try:
        return parse_fn(i)
    except Exception:
        return deft


parse_int = partial(parse_num, parse_fn=int, deft=-1)
parse_float = partial(parse_num, parse_fn=float, deft=-1.0)


def parse_devices_for_specific_list() -> List[int]:
    devicestr = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if not devicestr:
        warnings.warn("Warning: CUDA_VISIBLE_DEVICES is not set", RuntimeWarning)
        devices = []
    else:
        devices = [int(i.strip()) for i in devicestr.split(",")]
    return devices


def parse_gpustat_from_query(result: str) -> GpuStatus:
    parts = result.split(", ")
    # 'index', 'utilization.gpu', 'utilization.memory', 'temperature.gpu', 'temperature.memory', 'power.draw', 'power.limit', 'memory.total', 'memory.used', 'memory.free', 'memory.reserved'
    index = parse_int(parts[0])
    utilization_gpu = parse_int(parts[1])
    utilization_memory = parse_int(parts[2])
    temperature_gpu = parse_int(parts[3])
    temperature_memory = parse_int(parts[4])
    power_draw = parse_float(parts[5])
    power_limit = parse_float(parts[6])
    memory_total = parse_int(parts[7])
    memory_used = parse_int(parts[8])
    memory_free = parse_int(parts[9])
    memory_reserved = parse_int(parts[10])
    return GpuStatus(
        index,
        utilization=BaseRatio(utilization_gpu, utilization_memory),
        temperature=BaseRatio(temperature_gpu, temperature_memory),
        power=Power.from_cmd_watts(power_draw, power_limit),
        memory=Memory(memory_total, memory_used, memory_free, memory_reserved),
    )
