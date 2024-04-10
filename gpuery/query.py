import shutil
from typing import List, Optional
import subprocess

from .entity import GpuStatus, Power, Memory, Temperature, Utilization
from .parse import parse_gpustat_from_query

try:
    import pynvml as N

    NVML_ENABLED = True
except ImportError:
    NVML_ENABLED = False


def assert_cmd(cmd: str, install: Optional[str] = None):
    installation = f"Please install {install} first" if install else ""
    assert shutil.which(cmd) != None, f"{cmd} not found. {installation}"


def query_gpu_count(nvml: bool = NVML_ENABLED) -> int:
    if nvml:
        N.nvmlInit()
        count = N.nvmlDeviceGetCount()
        N.nvmlShutdown()
        return count
    else:
        assert_cmd("nvidia-smi", "nvidia-driver")
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.count("\n")


def query_gpus_with_nvml(index: List[int]) -> List[GpuStatus]:
    assert NVML_ENABLED, "run `pip install nvidia-ml-py` first to use this function"

    def query_gpu_by_index(idx: int) -> GpuStatus:
        handle = N.nvmlDeviceGetHandleByIndex(idx)
        device_index = N.nvmlDeviceGetIndex(handle)
        util = N.nvmlDeviceGetUtilizationRates(handle)
        utilization = Utilization(gpu=util.gpu, memory=util.memory)
        temperature = Temperature(
            gpu=N.nvmlDeviceGetTemperature(handle, N.NVML_TEMPERATURE_GPU),
            memory=-1,
            # memory=N.nvmlDeviceGetTemperature(handle, N.NVML_TEMPERATURE_MEM), # if N/A, an exception will be raised
        )
        power = Power.from_pkg_watts(
            draw=N.nvmlDeviceGetPowerUsage(handle),
            limit=N.nvmlDeviceGetPowerManagementLimit(handle),
        )

        mem: N.c_nvmlMemory_v2_t = N.nvmlDeviceGetMemoryInfo(handle, N.nvmlMemory_v2)
        memory = Memory.from_bytes(
            total=mem.total, used=mem.used, free=mem.free, reserved=mem.reserved
        )
        return GpuStatus(
            index=device_index,
            utilization=utilization,
            temperature=temperature,
            power=power,
            memory=memory,
        )

    N.nvmlInit()
    device_count = N.nvmlDeviceGetCount()
    for idx in index:
        assert 0 <= idx < device_count, f"Invalid GPU index: {idx}"
    gpus = [query_gpu_by_index(idx) for idx in index]
    N.nvmlShutdown()
    return gpus


QUERY_GPU_FIELDS = [
    "index",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "temperature.memory",
    "power.draw",
    "power.limit",
    "memory.total",
    "memory.used",
    "memory.free",
    "memory.reserved",
]
QUERY_FORMAT_FILEDS = ["csv", "noheader", "nounits"]
EXEC_CMD_LIST = [
    "nvidia-smi",
    f"--query-gpu={','.join(QUERY_GPU_FIELDS)}",
    f"--format={','.join(QUERY_FORMAT_FILEDS)}",
]


def query_gpus_with_smi(index: List[int]) -> List[GpuStatus]:
    """query gpu information through `nvidia-smi` command.
    :param index: a list of gpu index to query. CUDA_DEVICE_ORDER=PCI_BUS_ID
    """
    assert_cmd("nvidia-smi", "nvidia-driver")
    result = subprocess.run(
        EXEC_CMD_LIST, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.stderr:
        raise Exception(result.stderr)
    # result.stdout: "0, 0, 0, 48, N/A, 4.50, 80.00, 6144, 6, 5930, 207\n"
    queries = result.stdout.strip().split("\n")
    gpu_count = len(queries)
    for idx in index:
        assert 0 <= idx < gpu_count, f"Invalid GPU index: {idx}"
    gpus = [parse_gpustat_from_query(query) for query in queries]
    return [gpus[idx] for idx in index]


def query_gpus_auto(index: List[int]) -> List[GpuStatus]:
    if NVML_ENABLED:
        return query_gpus_with_nvml(index)
    else:
        return query_gpus_with_smi(index)
