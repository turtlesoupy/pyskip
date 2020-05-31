import gc
import pandas as pd
import time
import numpy as np
import functools
import torch
import torch.nn.functional
import _skimpy_cpp_ext
from _skimpy_bench_cpp_ext import taco, memory, run_length_array
from skimpy.config import num_threads_scope, set_value, config_scope, lazy_evaluation_scope, greedy_evaluation_scope, get_all_values
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

    def run(self, repeats=3):
        ret = defaultdict(list)
        for i in range(repeats):
            if hasattr(self, 'run_skimpy'):
                ret["skimpy"].append(self.run_skimpy(**self.suite_kwargs.get("skimpy", {})))
            if hasattr(self, 'run_skimpy_accelerated'):
                ret["skimpy_accelerated"].append(self.run_skimpy_accelerated(**self.suite_kwargs.get("skimpy_accelerated", {})))
            if hasattr(self, 'run_skimpy_lazy'):
                ret["skimpy_lazy"].append(self.run_skimpy_lazy(**self.suite_kwargs.get("skimpy_lazy", {})))
            if hasattr(self, 'run_skimpy_greedy'):
                ret["skimpy_greedy"].append(self.run_skimpy_greedy(**self.suite_kwargs.get("skimpy_greedy", {})))
            if hasattr(self, 'run_skimpy_slow'):
                ret["skimpy_slow"].append(self.run_skimpy_slow(**self.suite_kwargs.get("skimpy_slow", {})))
            if hasattr(self, 'run_taco'):
                ret["taco"].append(self.run_taco(**self.suite_kwargs.get("taco", {})))
            if hasattr(self, 'run_memory'):
                ret["memory"].append(self.run_memory(**self.suite_kwargs.get("memory", {})))
            if hasattr(self, 'run_numpy'):
                ret["numpy"].append(self.run_numpy(**self.suite_kwargs.get("numpy", {})))
            if hasattr(self, 'run_torch'):
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

    @torch.no_grad()
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

    @torch.no_grad()
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

    @torch.no_grad()
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


class DenseSkimpyImplementationBenchmark(Benchmark):
    # Tournament tree Vs. min
    # Hashtable Vs. no hash-table
    # Maybe a lazyness one (greedy evaluation)
    def __init__(self, array_length, num_inputs, suite_kwargs=None):
        self.array_length = array_length
        self.num_inputs = num_inputs
        self.suite_kwargs = suite_kwargs or {}

    def _numpy_inputs(self):
        return [np.random.randint(2**3, size=self.array_length, dtype=np.int32) for _ in range(self.num_inputs)]
    
    def _run_skimpy(self):
        inputs = [_skimpy_cpp_ext.from_numpy(t) for t in self._numpy_inputs()]

        t = Timer()
        with t:
            _ = functools.reduce(lambda x, y: x + y, inputs).eval()

        return t.duration_ms

    def run_skimpy_lazy(self):
        with lazy_evaluation_scope():
            return self._run_skimpy()

    def run_skimpy_greedy(self):
        with greedy_evaluation_scope():
            return self._run_skimpy()

    def run_skimpy_slow(self):
        with config_scope():
            set_value("accelerated_eval", False)
            return self._run_skimpy()

    def run_skimpy_accelerated(self):
        with config_scope():
            set_value("accelerated_eval", True)
            return self._run_skimpy()


class MinecraftConvolutionBenchmark(Benchmark):
    def __init__(self, level, kernel_width, suite_kwargs=None):
        self.level = level
        self.megatensor = level.megatensor().eval()
        self.kernel_width = kernel_width
        self.kernel_shape = tuple([kernel_width] * 3)
        self.suite_kwargs = suite_kwargs or {}
        self.numpy_chunk_list = [e.tensor.to_numpy() for e in level.chunk_list]

    def _numpy_kernel(self, dtype=np.int32):
        return np.random.randint(2**3, size=self.kernel_shape, dtype=np.int32).astype(dtype)

    def run_skimpy(self, num_threads=1):
        kernel = skimpy.Tensor.from_numpy(self._numpy_kernel())

        with num_threads_scope(num_threads):
            t = Timer()
            with t:
                _ = skimpy.convolve.conv_3d(self.megatensor, kernel).eval()
            return t.duration_ms

    @torch.no_grad()
    def run_torch(self, device="cpu"):
        dtype = np.int32 if device == "cpu" else np.float32
        kernel = torch.from_numpy(self._numpy_kernel(dtype=dtype)).to(device).reshape((1, 1) + self.kernel_shape)
        torch_operands = [
            torch.from_numpy(e).to(device).reshape((1, 1) + e.shape)
            for e in self.numpy_chunk_list
        ]

        t = Timer()
        with t:
            for operand in torch_operands:
                ret = torch.nn.functional.conv3d(operand, kernel).cpu()
                # Force torch to materialize if it is on the GPU
                tst = ret[0, 0, 0, 0, 0].item()
        return t.duration_ms

    def run_memory(self):
        return memory.no_simd_int_cum_sum_write(
            num_elements=len(self.megatensor), 
            num_input_arrays=1,
            num_threads=4,
        ) * MICR_TO_MS



class MNISTConvolutionBenchmark(Benchmark):
    def __init__(self, mnist_path, kernel_width, num_kernels, quantize_buckets=None, suite_kwargs=None):
        self.mnist_np_array = self.__class__._numpy_from_mnist(mnist_path, quantize_buckets=quantize_buckets)
        self.kernel_width = kernel_width
        self.kernel_shape = tuple([kernel_width] * 2)
        self.num_kernels = num_kernels
        self.suite_kwargs = suite_kwargs or {}

    @classmethod
    def _numpy_from_mnist(cls, path, quantize_buckets):
        csv = pd.read_csv(path)
        data_columns = [e for e in csv.columns if e != "label"]
        np_mnist = csv[data_columns].to_numpy()
        batch_size, example_size = np_mnist.shape
        digit_size = 28
        layout = np_mnist.reshape(digit_size, digit_size * batch_size).astype(np.int32)
        if quantize_buckets is None:
            return layout
        else:
            rng = 256 / quantize_buckets
            for i in range(quantize_buckets):
                start = i * rng  
                end = (i + 1) * rng  
                layout[(layout < end) & (layout >= start)] = i * quantize_buckets
        return layout

    def _numpy_kernels(self, dtype=np.int32):
        return [
            np.random.randint(2**3, size=self.kernel_shape, dtype=np.int32).astype(dtype)
            for _ in range(self.num_kernels)
        ]

    def run_skimpy(self, num_threads=1):
        operand = skimpy.Tensor.from_numpy(self.mnist_np_array)
        kernels = [skimpy.Tensor.from_numpy(e) for e in self._numpy_kernels()]

        with num_threads_scope(num_threads):
            t = Timer()
            with t:
                for kernel in kernels:
                    _ = skimpy.convolve.conv_2d(operand, kernel).eval()
            return t.duration_ms

    def run_torch(self, device="cpu"):
        operand = torch.from_numpy(self.mnist_np_array).to(device).reshape((1, 1) + self.mnist_np_array.shape)
        kernels = [torch.from_numpy(e).to(device).reshape((1, 1) + self.kernel_shape) for e in self._numpy_kernels()]

        t = Timer()
        with t:
            for kernel in kernels:
                ret = torch.nn.functional.conv2d(operand, kernel).cpu()
            # Force torch to materialize if it is on the GPU
            tst = ret[0, 0, 0, 0].item()
        return t.duration_ms

    def run_memory(self):
        return memory.no_simd_int_cum_sum_write(
            num_elements=self.mnist_np_array.size,  # Sparse is assumed to be a list of (pos, val)
            num_input_arrays=1,
            num_threads=4,
        ) * MICR_TO_MS
