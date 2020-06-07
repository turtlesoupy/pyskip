import gc
import asyncio
import pandas as pd
import time
import numpy as np
import functools
import sys
import torch
import torch.sparse
import torch.nn.functional
import _skimpy_cpp_ext
from _skimpy_bench_cpp_ext import memory, run_length_array
from skimpy.config import (
    num_threads_scope, set_value, config_scope, lazy_evaluation_scope, greedy_evaluation_scope
)
import skimpy
import skimpy.convolve
import scipy.signal
from contextlib import contextmanager
from collections import defaultdict

try:
    from _skimpy_bench_cpp_ext import taco
    TACO_ENABLED = True
except ImportError:
    print("Warning: disabling TACO support due to import error", file=sys.stderr)
    TACO_ENABLED = False

NANO_TO_MS = 1.0 / 1000000
MICR_TO_MS = 1.0 / 1000

@contextmanager
def torch_thread_scope(num_threads):
    restore_num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    assert torch.get_num_threads() == num_threads
    try:
        yield
    finally:
        torch.set_num_threads(restore_num_threads)


def torch_to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


class cached_property(object):
    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self

        if asyncio and asyncio.iscoroutinefunction(self.func):
            return self._wrap_in_coroutine(obj)

        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value

    def _wrap_in_coroutine(self, obj):
        @functools.wraps(obj)
        @asyncio.coroutine
        def wrapper():
            future = asyncio.ensure_future(self.func(obj))
            obj.__dict__[self.func.__name__] = future
            return future

        return wrapper()


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
    def run_suite_axis(cls, varying_arg_name, varying_arg, repeats, suite, **other_args):
        ret = {k: {k1: v1 for k1, v1 in v.items() if k1 not in ("kwargs", "method")} for k, v in suite.items()}
        for val in varying_arg:
            args = {varying_arg_name: val, **other_args}
            klass = cls(**args)
            for k, v in suite.items():
                kwargs = v.get("kwargs", {})
                method = v["method"]
                tot = 0
                for i in range(repeats):
                    tot += getattr(klass, method)(**kwargs)
                out = tot / repeats

                if "vals" not in ret[k]:
                    ret[k]["vals"] = [out]
                else:
                    ret[k]["vals"].append(out)
        
        for k in ret.keys():
            ret[k]["vals"] = np.array(ret[k]['vals'])
        
        return ret

    @classmethod
    def run_suite(cls, repeats, suite, **other_args):
        ret = {k: {k1: v1 for k1, v1 in v.items() if k1 not in ("kwargs", "method")} for k, v in suite.items()}

        klass = cls(**other_args)
        for k, v in suite.items():
            kwargs = v.get("kwargs", {})
            method = v["method"]
            tot = 0
            for i in range(repeats):
                tot += getattr(klass, method)(**kwargs)
            out = tot / repeats
            ret[k]["val"] = out
        
        return ret


    @classmethod
    def run_against_axis(cls, varying_arg_name, varying_arg, repeats, **other_args):
        ret = {}
        for val in varying_arg:
            args = {varying_arg_name: val, **other_args}
            times = cls(**args).run(repeats=repeats)
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
                ret["skimpy_accelerated"].append(
                    self.run_skimpy_accelerated(**self.suite_kwargs.get("skimpy_accelerated", {}))
                )
            if hasattr(self, 'run_skimpy_lazy'):
                ret["skimpy_lazy"].append(self.run_skimpy_lazy(**self.suite_kwargs.get("skimpy_lazy", {})))
            if hasattr(self, 'run_skimpy_greedy'):
                ret["skimpy_greedy"].append(self.run_skimpy_greedy(**self.suite_kwargs.get("skimpy_greedy", {})))
            if hasattr(self, 'run_skimpy_slow'):
                ret["skimpy_slow"].append(self.run_skimpy_slow(**self.suite_kwargs.get("skimpy_slow", {})))
            if hasattr(self, 'run_taco') and TACO_ENABLED:
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

    @cached_property
    def _numpy_inputs(self):
        return [np.random.randint(2**3, size=self.array_length, dtype=np.int32) for _ in range(self.num_inputs)]

    def run_numpy(self):
        inputs = self._numpy_inputs
        gc.collect()

        t = Timer()
        with t:
            _ = functools.reduce(lambda x, y: x + y, inputs)

        return t.duration_ms

    @torch.no_grad()
    def run_torch(self, num_threads=1):
        with torch_thread_scope(num_threads):
            inputs = [torch.from_numpy(t).cpu() for t in self._numpy_inputs]
            gc.collect()

            t = Timer()
            with t:
                _ = functools.reduce(lambda x, y: x + y, inputs)
        return t.duration_ms

    def run_taco(self):
        return taco.dense_sum(
            num_elements=self.array_length, num_input_arrays=self.num_inputs, include_compile_time=False
        ) * MICR_TO_MS

    def run_skimpy(self, num_threads=1, use_custom_kernel=False):
        inputs = [_skimpy_cpp_ext.from_numpy(t) for t in self._numpy_inputs]
        gc.collect()

        with num_threads_scope(num_threads):
            set_value("accelerated_eval", False)
            if use_custom_kernel:
                set_value("custom_eval_kernel", "add_int32")
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

    @cached_property
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
    
    @cached_property
    def _skimpy_inputs(self):
        return [_skimpy_cpp_ext.from_numpy(t) for t in self._numpy_inputs]

    def run_numpy(self):
        inputs = self._numpy_inputs
        gc.collect()
        t = Timer()
        with t:
            b = functools.reduce(lambda x, y: x + y, inputs)
            print(b[-1])
        return t.duration_ms

    def run_skimpy(self, num_threads=1, use_custom_kernel=False):
        inputs = self._skimpy_inputs
        gc.collect()

        with num_threads_scope(num_threads):
            set_value("accelerated_eval", False) # TODO: check me
            if use_custom_kernel:
                set_value("custom_eval_kernel", "add_int32")
            t = Timer()
            with t:
                _ = functools.reduce(lambda x, y: x + y, inputs).eval()

        return t.duration_ms

    def run_torch(self, num_threads=1, dense=True):
        with torch_thread_scope(num_threads):
            if dense:
                inputs = [torch.from_numpy(t).cpu() for t in self._numpy_inputs]
            else:
                inputs = [torch_to_sparse(torch.from_numpy(t)).cpu() for t in self._numpy_inputs]

            t = Timer()
            with t:
                _ = functools.reduce(lambda x, y: x + y, inputs)
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
    def run_torch(self, device="cpu", num_threads=1):
        dtype = np.int32 if device == "cpu" else np.float32
        operand = torch.from_numpy(self._numpy_input(dtype=dtype)).to(device).reshape((1, 1) + self.shape)
        kernel = torch.from_numpy(self._numpy_kernel(dtype=dtype)).to(device).reshape((1, 1) + self.kernel_shape)

        with torch_thread_scope(num_threads):
            t = Timer()
            with t:
                ret = torch.nn.functional.conv3d(operand, kernel).cpu()
                # Force torch to materialize if it is on the GPU
                tst = ret[0, 0, 0, 0, 0].item()
            return t.duration_ms

    def run_numpy(self):
        operand = self._numpy_input()
        kernel = self._numpy_kernel()
        t = Timer()
        with t:
            scipy.signal.convolve(operand, kernel)

        return t.duration_ms

    def run_skimpy(self, num_threads=1):
        operand = skimpy.Tensor.from_numpy(self._numpy_input())
        kernel = skimpy.Tensor.from_numpy(self._numpy_kernel())

        with num_threads_scope(num_threads):
            set_value("accelerated_eval", False) # TODO: check me
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
    def __init__(
        self,
        shape,
        kernel_width,
        num_non_zero,
        run_length,
        align_inputs,
        deterministic_run_length,
        suite_kwargs=None
    ):
        assert len(shape) == 3
        assert kernel_width % 2 == 1

        self.shape = shape
        self.kernel_shape = tuple([kernel_width] * 3)

        self.array_length = functools.reduce(lambda x, y: x * y, shape)
        self.num_non_zero = num_non_zero
        self.run_length = run_length
        self.align_inputs = align_inputs
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
            set_value("accelerated_eval", False) # TODO: check me
            t = Timer()
            with t:
                _ = skimpy.convolve.conv_3d(operand, kernel).eval()
            return t.duration_ms

    def run_numpy(self):
        operand = self._numpy_input()
        kernel = self._numpy_kernel()
        t = Timer()
        with t:
            scipy.signal.convolve(operand, kernel)

        return t.duration_ms

    @torch.no_grad()
    def run_torch(self, device="cpu", num_threads=1):
        with torch_thread_scope(num_threads):
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
            num_input_arrays=1,
            num_threads=4,
        ) * MICR_TO_MS


class DenseSkimpyImplementationBenchmark(Benchmark):
    def __init__(self, array_length, num_inputs, suite_kwargs=None):
        self.array_length = array_length
        self.num_inputs = num_inputs
        self.suite_kwargs = suite_kwargs or {}

    def _numpy_inputs(self):
        return [np.random.randint(2**3, size=self.array_length, dtype=np.int32) for _ in range(self.num_inputs)]

    def _run_skimpy(self):
        inputs = [_skimpy_cpp_ext.from_numpy(t) for t in self._numpy_inputs()]
        gc.collect()

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


class RunLengthSkimpyImplementationBenchmark(Benchmark):
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

    def _run_skimpy(self):
        inputs = [_skimpy_cpp_ext.from_numpy(t) for t in self._numpy_inputs()]
        gc.collect()

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
            set_value("accelerated_eval", False) # TODO: check me
            t = Timer()
            with t:
                _ = skimpy.convolve.conv_3d(self.megatensor, kernel).eval()
            return t.duration_ms

    def run_numpy(self, use_mt=False):
        if use_mt:
            operands = [self.megatensor.to_numpy()]
        else:
            operands = [torch.from_numpy(e) for e in self.numpy_chunk_list]

        kernel = self._numpy_kernel()
        t = Timer()
        with t:
            for operand in operands:
                scipy.signal.convolve(operand, kernel)

        return t.duration_ms


    @torch.no_grad()
    def run_torch(self, device="cpu", num_threads=1, use_mt=False):
        with torch_thread_scope(num_threads):
            dtype = np.int32 if device == "cpu" else np.float32
            kernel = torch.from_numpy(self._numpy_kernel(dtype=dtype)).to(device).reshape((1, 1) + self.kernel_shape)
            if use_mt:
                np_mt = self.megatensor.to_numpy()
                torch_operands = [torch.from_numpy(np_mt.astype(dtype)).to(device).reshape((1, 1) + np_mt.shape)]
            else:
                torch_operands = [torch.from_numpy(e.astype(dtype)).to(device).reshape((1, 1) + e.shape) for e in self.numpy_chunk_list]

            rets = []
            t = Timer()
            with t:
                for operand in torch_operands:
                    rets.append(torch.nn.functional.conv3d(operand, kernel))
                for ret in rets:
                    # Force torch to materialize if it is on the GPU
                    tst = ret[0, 0, 0, 0, 0].cpu().item()
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

    def run_numpy(self, use_mt=False):
        operand = self.mnist_np_array
        kernels = self._numpy_kernels()

        t = Timer()
        with t:
            for kernel in kernels:
                scipy.signal.convolve(operand, kernel)

        return t.duration_ms

    def run_skimpy(self, num_threads=1):
        operand = skimpy.Tensor.from_numpy(self.mnist_np_array)
        kernels = [skimpy.Tensor.from_numpy(e) for e in self._numpy_kernels()]

        with num_threads_scope(num_threads):
            set_value("accelerated_eval", False) # TODO: check me
            t = Timer()
            with t:
                for kernel in kernels:
                    _ = skimpy.convolve.conv_2d(operand, kernel).eval()
            return t.duration_ms

    @torch.no_grad()
    def run_torch(self, device="cpu", num_threads=1):
        with torch_thread_scope(num_threads):
            dtype = np.int32 if device == "cpu" else np.float32
            operand = torch.from_numpy(self.mnist_np_array.astype(dtype)).to(device).reshape((1, 1) + self.mnist_np_array.shape)
            kernels = [torch.from_numpy(e.astype(dtype)).to(device).reshape((1, 1) + self.kernel_shape) for e in self._numpy_kernels()]

            rets = []
            t = Timer()
            with t:
                for kernel in kernels:
                    rets.append(torch.nn.functional.conv2d(operand, kernel))
                for ret in rets:
                    # Force torch to materialize if it is on the GPU
                    tst = ret[0, 0, 0, 0].cpu().item()
        return t.duration_ms

    def run_memory(self):
        return memory.no_simd_int_cum_sum_write(
            num_elements=self.mnist_np_array.size,  # Sparse is assumed to be a list of (pos, val)
            num_input_arrays=1,
            num_threads=4,
        ) * MICR_TO_MS
