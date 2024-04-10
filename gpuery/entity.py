from dataclasses import dataclass
from typing import Union, Callable
from functools import partial


@dataclass
class BaseRatio:  # unit: percentage(%)
    gpu: int  # utilization.gpu/temperature.gpu
    memory: int  # utilization.memory/temperature.memory
Utilization = BaseRatio  # alias for BaseRatio
Temperature = BaseRatio  # alias for BaseRatio

MiB = 2**20  # MiB = 2^20 bytes


def radix_conv(value: int, quantity: int, precision: int = 1) -> Union[int, float]:
    assert 6 >= precision >= 0, "precision should be between 0 and 6"
    assert quantity > 0, "quantity should be greater than 0"
    if precision == 0:  # int
        return value // quantity
    else:  # float
        return round(value / quantity, precision)


@dataclass
class Memory:  # unit: MiB [nvidia-smi]
    total: int  # memory.total
    used: int
    free: int
    reserved: int

    RADIX = partial(radix_conv, quantity=MiB, precision=0)

    @classmethod
    def from_bytes(
        cls: "Memory", total: int, used: int, free: int, reserved: int
    ) -> "Memory":
        return Memory(
            cls.RADIX(total), cls.RADIX(used), cls.RADIX(free), cls.RADIX(reserved)
        )


@dataclass
class Power:  # unit: Watts [nvidia-smi]
    draw: int  # decimal scaled value, like 46 (Watts * scale)
    limit: int  # the same as above
    scale: int = 10 # scale ratio

    RADIX_PKG = partial(radix_conv, quantity=100, precision=0)
    RADIX_CMD: Callable[[float], int] = lambda x: int(x*10)

    @classmethod
    def from_pkg_watts(cls: "Power", draw: float, limit: float) -> "Power":
        return Power(cls.RADIX_PKG(draw), cls.RADIX_PKG(limit))
    
    @classmethod
    def from_cmd_watts(cls: "Power", draw: float, limit: float) -> "Power":
        return Power(cls.RADIX_CMD(draw), cls.RADIX_CMD(limit))


@dataclass
class GpuStatus:
    index: int
    utilization: Utilization  # utilization block
    temperature: Temperature  # temperature block
    power: Power
    memory: Memory


@dataclass
class Threshold:
    mem: float = .1
    util: float = .1
    power: float = .1 + .1 # Standby + Usage (percentage)
    temp: int = 40
