import os
from typing import List, Callable
import warnings

from .entity import Threshold, GpuStatus
from .query import query_gpus_auto, query_gpu_count

def pass_threshold(thres: Threshold, status: GpuStatus) -> bool:
    # 1. mem
    if status.memory.used / status.memory.total > thres.mem:
        return False
    # 2. util
    if status.utilization.gpu/100 > thres.util:
        return False
    if status.utilization.memory/100 > thres.util:
        return False
    # 3. power
    if status.power.draw / status.power.limit > thres.power:
        return False
    # 4. temp
    if status.temperature.gpu > thres.temp:
        return False
    return True

def get_gpu_availability(device: int, threshold: Threshold, checker: Callable[[Threshold, GpuStatus], bool]=pass_threshold) -> bool:
    status = query_gpus_auto([device])[0]
    return checker(threshold, status)

def get_available_gpu_indexes(devices: List[int], threshold: Threshold, checker: Callable[[Threshold, GpuStatus], bool]=pass_threshold) -> List[int]:
    """
    index is ordered by pci_bus_id
    :warning: make sure U use pci_bus_id to identify GPUs, instead of performance by default.
    """
    status = query_gpus_auto(devices)
    return [d for d, s in zip(devices, status) if checker(threshold, s)]

DefaultThreshold = Threshold()
def retrive(threshold: Threshold = DefaultThreshold) -> List[int]:
    """retrive available gpu indexes depending on CUDA_VISIBLE_DEVICES and devices count.
    """
    order = os.getenv("CUDA_DEVICE_ORDER")
    if order != "PCI_BUS_ID":
        warnings.warn("\033[0;33mWarning:\033[0m CUDA_DEVICE_ORDER is not set to PCI_BUS_ID. This may cause unexpected behavior.", RuntimeWarning)
    devices = os.getenv("CUDA_VISIBLE_DEVICES", list(range(query_gpu_count())))
    if isinstance(devices, str):
        devices = [int(d) for d in devices.split(",")]

    return get_available_gpu_indexes(devices, threshold)


if __name__ == "__main__":
    print(retrive(threshold=Threshold(mem=0.1, util=0.3, power=0.1, temp=80)))
    print(retrive())