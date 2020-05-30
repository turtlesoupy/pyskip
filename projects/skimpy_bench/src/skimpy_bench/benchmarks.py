import gc
import time
import numpy as np
import random
import functools
import torch
import torch.nn.functional
import _skimpy_cpp_ext
from _skimpy_bench_cpp_ext import taco, memory, run_length_array
from tqdm.auto import tqdm
from skimpy.config import num_threads_scope
import skimpy
import skimpy.convolve
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
            args = {varying_arg_name: val, **other_args}
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
            if hasattr(self, 'run_skimpy'):
                if verbose:
                    print(f"{self.__class__.__name__} running skimpy")
                ret["skimpy"].append(self.run_skimpy(**self.suite_kwargs.get("skimpy", {})))
            if hasattr(self, 'run_taco'):
                if verbose:
                    print(f"{self.__class__.__name__} running taco")
                ret["taco"].append(self.run_taco(**self.suite_kwargs.get("taco", {})))
            if hasattr(self, 'run_memory'):
                if verbose:
                    print(f"{self.__class__.__name__} running memory")
                ret["memory"].append(self.run_memory(**self.suite_kwargs.get("memory", {})))
            if hasattr(self, 'run_numpy'):
                if verbose:
                    print(f"{self.__class__.__name__} running numpy")
                ret["numpy"].append(self.run_numpy(**self.suite_kwargs.get("numpy", {})))
            if hasattr(self, 'run_torch'):
                if verbose:
                    print(f"{self.__class__.__name__} running torch")
                ret["torch"].append(self.run_torch(**self.suite_kwargs.get("torch", {})))
        return {k: np.mean(np.array(v)) for k, v in ret.items()}


class DenseArrayBenchmark(Benchmark):
    def __init__(self, array_length, num_inputs, suite_kwargs=None):
        self.array_length = array_length
        self.num_inputs = num_inputs
        self.suite_kwargs = suite_kwargs or {}

    def _numpy_inputs(self):
        return [np.random.randint(2**3, size=self.array_length, dtype=np.int32) for _ in range(self.num_inputs)]

    def run_numpy(self):
        inputs = self._numpy_inputs()

        t = Timer()
        with t:
            _ = functools.reduce(lambda x, y: x + y, inputs)

        return t.duration_ms

    def run_torch(self):
        inputs = [torch.from_numpy(t).cpu() for t in self._numpy_inputs()]

        t = Timer()
        with t:
            _ = functools.reduce(lambda x, y: x + y, inputs)
        return t.duration_ms

    def run_taco(self):
        return taco.dense_sum(
            num_elements=self.array_length, num_input_arrays=self.num_inputs, include_compile_time=False
        ) * MICR_TO_MS

    def run_skimpy(self, num_threads=1):
        inputs = [_skimpy_cpp_ext.from_numpy(t) for t in self._numpy_inputs()]

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


class RunLengthArrayBenchmark(Benchmark):
    def __init__(
        self,
        array_length,
        num_non_zero,
        run_length,
        align_inputs,
        num_inputs,
        deterministic_run_length,
        suite_kwargs=None
    ):
        self.array_length = array_length
        self.num_non_zero = num_non_zero
        self.run_length = run_length
        self.align_inputs = align_inputs
        self.num_inputs = num_inputs
        self.deterministic_run_length = deterministic_run_length
        self.suite_kwargs = suite_kwargs or {}

    def _numpy_inputs(self):
        inputs = []
        seed = 42
        for _ in range(self.num_inputs):
            arr = run_length_array(
                num_elements=self.array_length,
                num_non_zero=self.num_non_zero,
                run_length=self.run_length,
                deterministic_run_length=self.deterministic_run_length,
                random_seed=seed
            )
            inputs.append(arr)
            if not self.align_inputs:
                seed += 1

        return inputs

    def run_numpy(self):
        inputs = self._numpy_inputs()
        t = Timer()
        with t:
            _ = functools.reduce(lambda x, y: x + y, inputs)
        return t.duration_ms

    def run_skimpy(self, num_threads=1):
        inputs = [_skimpy_cpp_ext.from_numpy(t) for t in self._numpy_inputs()]

        with num_threads_scope(num_threads):
            t = Timer()
            with t:
                _ = functools.reduce(lambda x, y: x + y, inputs).eval()

        return t.duration_ms

    def run_taco(self):
        return taco.sparse_sum(
            num_elements=self.array_length,
            num_non_zero=self.num_non_zero,
            run_length=self.run_length,
            align_inputs=self.align_inputs,
            num_input_arrays=self.num_inputs,
            deterministic_run_length=self.deterministic_run_length,
            include_compile_time=False,
        ) * MICR_TO_MS

    def run_memory(self):
        return memory.no_simd_int_cum_sum_write(
            num_elements=self.num_non_zero * 2,  # Sparse is assumed to be a list of (pos, val)
            num_input_arrays=self.num_inputs,
            num_threads=4,
        ) * MICR_TO_MS


class Dense3DConvolutionBenchmark(Benchmark):
    def __init__(self, shape, kernel_width, suite_kwargs=None):
        assert len(shape) == 3
        assert kernel_width % 2 == 1

        self.shape = shape
        self.kernel_shape = tuple([kernel_width] * 3)
        self.array_length = functools.reduce(lambda x, y: x * y, shape)
        self.suite_kwargs = suite_kwargs or {}

    def _numpy_input(self, dtype=np.int32):
        return np.random.randint(2**3, size=self.shape, dtype=np.int32).astype(dtype)

    def _numpy_kernel(self, dtype=np.int32):
        return np.random.randint(2**3, size=self.kernel_shape, dtype=np.int32).astype(dtype)

    def run_torch(self, device="cpu"):
        dtype = np.int32 if device == "cpu" else np.float32
        operand = torch.from_numpy(self._numpy_input(dtype=dtype)).to(device).reshape((1, 1) + self.shape)
        kernel = torch.from_numpy(self._numpy_kernel(dtype=dtype)).to(device).reshape((1, 1) + self.kernel_shape)

        t = Timer()
        with t:
            ret = torch.nn.functional.conv3d(operand, kernel).cpu()
            # Force torch to materialize if it is on the GPU
            tst = ret[0, 0, 0, 0, 0].item()
        return t.duration_ms

    def run_skimpy(self, num_threads=1):
        operand = skimpy.Tensor.from_numpy(self._numpy_input())
        kernel = skimpy.Tensor.from_numpy(self._numpy_kernel())

        with num_threads_scope(num_threads):
            t = Timer()
            with t:
                _ = skimpy.convolve.conv_3d(operand, kernel).eval()
            return t.duration_ms

    def run_memory(self):
        return memory.no_simd_int_cum_sum_write(
            num_elements=self.array_length,
            num_input_arrays=1,
            num_threads=4,
        ) * MICR_TO_MS


class RunLength3DConvolutionBenchmark(Benchmark):
    def __init__(self, shape, kernel_width, num_non_zero, run_length, align_inputs, num_inputs, deterministic_run_length, suite_kwargs=None):
        assert len(shape) == 3
        assert kernel_width % 2 == 1

        self.shape = shape
        self.kernel_shape = tuple([kernel_width] * 3)

        self.array_length = functools.reduce(lambda x, y: x * y, shape)
        self.num_non_zero = num_non_zero
        self.run_length = run_length
        self.align_inputs = align_inputs
        self.num_inputs = num_inputs
        self.deterministic_run_length = deterministic_run_length
        self.suite_kwargs = suite_kwargs or {}

    def _numpy_input(self, dtype=np.int32):
        return run_length_array(
            num_elements=self.array_length,
            num_non_zero=self.num_non_zero,
            run_length=self.run_length,
            deterministic_run_length=self.deterministic_run_length,
            random_seed=42,
        ).reshape(self.shape).astype(dtype)

    def _numpy_kernel(self, dtype=np.int32):
        return np.random.randint(2**3, size=self.kernel_shape, dtype=np.int32).astype(dtype)

    def run_skimpy(self, num_threads=1):
        operand = skimpy.Tensor.from_numpy(self._numpy_input())
        kernel = skimpy.Tensor.from_numpy(self._numpy_kernel())

        with num_threads_scope(num_threads):
            t = Timer()
            with t:
                _ = skimpy.convolve.conv_3d(operand, kernel).eval()
            return t.duration_ms

    def run_torch(self, device="cpu"):
        dtype = np.int32 if device == "cpu" else np.float32
        operand = torch.from_numpy(self._numpy_input(dtype=dtype)).to(device).reshape((1, 1) + self.shape)
        kernel = torch.from_numpy(self._numpy_kernel(dtype=dtype)).to(device).reshape((1, 1) + self.kernel_shape)

        t = Timer()
        with t:
            ret = torch.nn.functional.conv3d(operand, kernel).cpu()
            # Force torch to materialize if it is on the GPU
            tst = ret[0, 0, 0, 0, 0].item()
        return t.duration_ms

    def run_memory(self):
        return memory.no_simd_int_cum_sum_write(
            num_elements=self.num_non_zero * 2,  # Sparse is assumed to be a list of (pos, val)
            num_input_arrays=self.num_inputs,
            num_threads=4,
        ) * MICR_TO_MS


class SkimpyImplementationBenchmark:
    # Tournament tree Vs. min
    # Hashtable Vs. no hash-table
    # Maybe a lazyness one (greedy evaluation)
    pass


class MinecraftConvolutionBenchmark:
    pass


class MNISTConvolutionBenchmark:
    pass
