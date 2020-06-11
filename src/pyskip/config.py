from contextlib import contextmanager
from _pyskip_cpp_ext import config as _pyskip_config

from . exceptions import TypeConversionError


@contextmanager
def config_scope(**kwargs):
    vals = _pyskip_config.get_all_values()
    try:
        if kwargs:
            for k, v in kwargs.items():
                set_value(k, v)
        yield
    finally:
        _pyskip_config.set_all_values(vals)


@contextmanager
def num_threads_scope(num):
    with config_scope():
        _pyskip_config.set_int_value("parallelize_parts", num)
        yield


@contextmanager
def flush_threshold_scope(num):
    with config_scope():
        _pyskip_config.set_int_value("flush_tree_size_threshold", num)
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
    return _pyskip_config.get_all_values()


def set_value(name, val):
    if isinstance(val, bool):
        _pyskip_config.set_bool_value(name, val)
    elif isinstance(val, int):
        _pyskip_config.set_int_value(name, val)
    elif isinstance(val, float):
        _pyskip_config.set_float_value(name, val)
    elif isinstance(val, str):
        _pyskip_config.set_string_value(name, val)
    else:
        raise TypeConversionError(f"Unsupported type for config map: {type(val)}")