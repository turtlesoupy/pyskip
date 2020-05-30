import gc
import time
import numpy as np
import random
import functools
import _skimpy_cpp_ext
from _skimpy_bench_cpp_ext import taco, memory
from skimpy.config import num_threads_scope
from collections import defaultdict
from contextlib import contextmanager

NANO_TO_MS = 1.0 / 1000000
MICR_TO_MS = 1.0 / 1000


class Timer:
    def __init__(self):
        self.duration_ns = None
        self.start_ns = None

    def __enter__(self, *args):
        gc.disable()
        self.start_ns = time.perf_counter_ns()

    def __exit__(self, typ, value, traceback):
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
    

class Benchmark:
    @classmethod
    def run_against_axis(cls, varying_arg_name, varying_arg, repeats, verbose=False, **other_args):
        ret = {}
        for val in varying_arg:
            args = {
                varying_arg_name: val,
                **other_args
            }
            times = cls(**args).run(repeats=repeats, verbose=verbose)
            for k, v in times.items():
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]
        return {k: np.array(v) for k, v in ret.items()}

    def run(self, repeats=3, verbose=False):
        ret = defaultdict(list)
        for i in range(repeats):
            if verbose:
                print(f"{self.__class__.__name__} running skimpy")
            ret["skimpy"].append(self.run_skimpy(**self.suite_kwargs.get("skimpy", {})))
            if verbose:
                print(f"{self.__class__.__name__} running taco")
            ret["taco"].append(self.run_taco(**self.suite_kwargs.get("taco", {})))
            if verbose:
                print(f"{self.__class__.__name__} running memory")
            ret["memory"].append(self.run_memory(**self.suite_kwargs.get("memory", {})))
            if verbose:
                print(f"{self.__class__.__name__} running numpy")
            ret["numpy"].append(self.run_numpy(**self.suite_kwargs.get("numpy", {})))
        return {k: np.mean(np.array(v)) for k, v in ret.items()}


class DenseArrayBenchmark(Benchmark):
    def __init__(self, array_length, num_inputs, suite_kwargs=None):
        self.array_length = array_length
        self.num_inputs = num_inputs
        self.suite_kwargs = suite_kwargs or {}

    def run_numpy(self):
        inputs = [np.random.randint(2**3, size=self.array_length) for _ in range(self.num_inputs)]

        t = Timer()
        with t:
            _ = functools.reduce(lambda x, y: x + y, inputs)

        return t.duration_ms

    def run_taco(self):
        return taco.dense_sum(
            num_elements=self.array_length, num_input_arrays=self.num_inputs, include_compile_time=False
        ) * MICR_TO_MS

    def run_skimpy(self, num_threads=1):
        inputs = [
            _skimpy_cpp_ext.from_numpy(np.random.randint(2**3, size=self.array_length)) for _ in range(self.num_inputs)
        ]

        with num_threads_scope(num_threads):
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


class DenseConvolutionBenchmark:
    pass


class RunLengthArrayBenchmark(Benchmark):
    def __init__(self, array_length, num_non_zero, max_run_length, align_inputs, num_inputs, suite_kwargs=None):
        self.array_length = array_length
        self.num_non_zero = num_non_zero
        self.max_run_length = max_run_length
        self.align_inputs = align_inputs
        self.num_inputs = num_inputs
        self.suite_kwargs = suite_kwargs or {}

    def run_numpy(self):
        inputs = [np.random.randint(2**3, size=self.array_length) for _ in range(self.num_inputs)]
        t = Timer()
        with t:
            _ = functools.reduce(lambda x, y: x + y, inputs)
        return t.duration_ms

    def run_skimpy(self, num_threads=1):
        inputs = []
        for i in range(self.num_inputs):
            builder = _skimpy_cpp_ext.IntBuilder(self.array_length, 0)
            if self.align_inputs:
                random.seed(42)

            for j in range(self.num_non_zero):
                insert_position = random.randint(0, self.array_length)
                run_length = random.randint(1, self.max_run_length)
                val = random.randint(0, 2 ** 16)
                builder[insert_position:(insert_position + run_length)] = val
            inputs.append(builder.build())

        with num_threads_scope(num_threads):
            t = Timer()
            with t:
                _ = functools.reduce(lambda x, y: x + y, inputs).eval()

        return t.duration_ms

    def run_taco(self):
        return taco.sparse_sum(
            num_elements=self.array_length,
            num_non_zero=self.num_non_zero,
            max_run_length=self.max_run_length,
            align_inputs=self.align_inputs,
            num_input_arrays=self.num_inputs,
            include_compile_time=False,
        ) * MICR_TO_MS

    def run_memory(self):
        return memory.no_simd_int_cum_sum_write(
            num_elements=self.num_non_zero * 2,  # Sparse is assumed to be a list of (pos, val)
            num_input_arrays=self.num_inputs,
            num_threads=4,
        ) * MICR_TO_MS


class SparseConvolutionBenchmark:
    pass


class RunLengthConvolutionBenchmark:
    pass


class MinecraftConvolutionBenchmark:
    pass


class MNISTConvolutionBenchmark:
    pass
