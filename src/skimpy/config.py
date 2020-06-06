from contextlib import contextmanager
from _skimpy_cpp_ext import config as _skimpy_config

from . exceptions import TypeConversionError


@contextmanager
def config_scope(**kwargs):
    vals = _skimpy_config.get_all_values()
    try:
        if kwargs:
            for k, v in kwargs.items():
                set_value(k, v)
        yield
    finally:
        _skimpy_config.set_all_values(vals)


@contextmanager
def num_threads_scope(num):
    with config_scope():
        _skimpy_config.set_int_value("parallelize_parts", num)
        yield


@contextmanager
def flush_threshold_scope(num):
    with config_scope():
        _skimpy_config.set_int_value("flush_tree_size_threshold", num)
        yield


@contextmanager
def lazy_evaluation_scope():
    with flush_threshold_scope(2 ** 30):
        yield


@contextmanager
def greedy_evaluation_scope():
    with flush_threshold_scope(0):
        yield


def get_all_values():
    return _skimpy_config.get_all_values()


def set_value(name, val):
    if isinstance(val, bool):
        _skimpy_config.set_bool_value(name, val)
    elif isinstance(val, int):
        _skimpy_config.set_int_value(name, val)
    elif isinstance(val, float):
        _skimpy_config.set_float_value(name, val)
    elif isinstance(val, str):
        _skimpy_config.set_string_value(name, val)
    else:
        raise TypeConversionError(f"Unsupported type for config map: {type(val)}")