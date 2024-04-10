import unittest

from gpuery.parse import parse_gpustat_from_query, parse_num
from gpuery.query import query_gpus_with_smi, query_gpus_with_nvml, query_gpu_count
from gpuery.select import retrive, get_available_gpu_indexes

class ParseTest(unittest.TestCase):
    def test_parse_num(self,):
        self.assertEqual(parse_num("123", int, 0), 123)
        self.assertEqual(parse_num("123n", int, -1), -1)
        self.assertEqual(parse_num("n/a", int, -1), -1)
        self.assertEqual(parse_num("N/A", int, -1), -1)
        self.assertEqual(parse_num("N/A", float, -1.), -1.)
        self.assertEqual(parse_num("233", float, -1.), 233.)
        self.assertEqual(parse_num("1.2", float, -1.), 1.2)
        self.assertEqual(parse_num("2e3", float, -1.), 2e3)

    def test_parse_gpustat(self,):
        from gpuery.entity import GpuStatus, Memory, Power, BaseRatio
        self.assertEqual(parse_gpustat_from_query("0, 0, 0, 45, N/A, 4.49, 80.00, 6144, 6, 5930, 207"), GpuStatus(
            index=0,
            utilization=BaseRatio(0,0),
            temperature=BaseRatio(45, -1),
            power=Power.from_cmd_watts(4.49, 80.0),
            memory=Memory(6144, 6, 5930, 207),
        ))

class QueryTest(unittest.TestCase):
    def test_query_gpus(self,):
        indexes = [0]
        nv = query_gpus_with_smi(indexes)
        pynv = query_gpus_with_nvml(indexes)
        self.assertEqual(nv, pynv) # fails sometimes

    def test_query_gpu_count(self,):
        try:
            import pynvml
            enabled = True
        except ImportError:
            enabled = False
        if enabled:
            self.assertEqual(query_gpu_count(True), query_gpu_count(False))

class SelectTest(unittest.TestCase):
    def test_select_gpus(self,):
        import os
        from gpuery.entity import Threshold
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        thre = Threshold()
        self.assertEqual(retrive(thre), get_available_gpu_indexes([0], thre))

if __name__ == "__main__":
    unittest.main()