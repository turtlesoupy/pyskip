import gc
import time
import numpy as np
import functools
import _skimpy_cpp_ext
from _skimpy_bench_cpp_ext import taco, memory
from collections import defaultdict

NANO_TO_MS = 1.0 / 1000000
MICR_TO_MS = 1.0 / 1000

class Timer:
    def __init__(self):
        self.duration_ns = None
        self.start_ns = None

    def __enter__(self, *args):
        gc.disable()
        self.start_ns = time.perf_counter_ns()

    def __exit__(self, type, value, traceback):
        self.duration_ns = time.perf_counter_ns() - self.start_ns
        gc.enable()

    @property
    def duration_ms(self):
        return self.duration_ns * NANO_TO_MS
        
    @classmethod
    def repeated_average(cls, code, count):
        total_time = 0
        for i in range(count):
            total_time += code()
        return total_time / count
    
    @classmethod
    def repeated_median(cls, code, count):
        times = []
        for i in range(count):
            times.append(code())
        return sorted(times)[len(times) // 2]


class DenseArrayBenchmark:
    def __init__(self, array_length, num_inputs):
        self.array_length = array_length
        self.num_inputs = num_inputs

    def run_numpy(self):
        inputs = [np.random.randint(2 ** 3, size=self.array_length) for _ in range(self.num_inputs)]

        t = Timer()
        with t:
            _ = functools.reduce(lambda x, y: x + y, inputs)

        return t.duration_ms

    def run_taco(self):
        return taco.dense_sum(
            num_elements=self.array_length,
            num_input_arrays=self.num_inputs,
            include_compile_time=False
        ) * MICR_TO_MS

    def run_skimpy(self):
        inputs = [_skimpy_cpp_ext.from_numpy(np.random.randint(2 ** 3, size=self.array_length)) for _ in range(self.num_inputs)]
        
        t = Timer()
        with t:
            _ = functools.reduce(lambda x, y: x + y, inputs).eval()
            
        return t.duration_ms
    
    def run_memory(self):
        return memory.no_simd_int_cum_sum_write(
            num_elements=self.array_length,
            num_input_arrays=self.num_inputs,
            num_threads=4,
        ) * MICR_TO_MS

    def run(self, repeats=3, **kwargs):
        ret = defaultdict(list)
        for i in range(repeats):
            ret["skimpy"].append(self.run_skimpy(**kwargs))
            ret["taco"].append(self.run_skimpy(**kwargs))
            ret["memory"].append(self.run_memory(**kwargs))
            ret["numpy"].append(self.run_numpy(**kwargs))
        return {k: np.mean(np.array(v)) for k, v in ret.items()}


class DenseConvolutionBenchmark:
    pass


class SparseArrayBenchmark:
    pass


class SparseConvolutionBenchmark:
    pass


class RunLengthArrayBenchmark:
    pass


class RunLengthConvolutionBenchmark:
    pass


class MinecraftConvolutionBenchmark:
    pass


class MNISTConvolutionBenchmark:
    pass
