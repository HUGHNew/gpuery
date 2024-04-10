# gpuery

> A simple GPU available querier on isolated node

## installation

`pip install gpuery`

or 

`pip install gpuery[nvml]`

There is no functional difference between the above two. The only difference is how to obtain the graphics card information. One uses `nvidia-smi` and the other uses `nvidia-ml-py`

## usage

```python
from typing import List
import gpuery

idle_cards:List[int] = gpuery.retrive() # get all idle cards in current machine using default threshold

idle_cards:List[int] = gpuery.get_available_gpu_indexes([0, 1]) # get available cards from cards 0,1 using default threshold

thres = gpuery.Threshold(
  mem: float = .1 # memory usage percentage
  util: float = .2 # utilization usage percentage
  power: float = .2 # power usage percentage
  temp: int = 40 # temperature limitation in Celsius
)

idle_cards:List[int] = gpuery.retrive(thres) # get all idle cards in current machine using specific threshold
```