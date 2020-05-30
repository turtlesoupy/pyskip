from contextlib import contextmanager
from _skimpy_cpp_ext import config as _skimpy_config


@contextmanager
def config_scope():
    vals = _skimpy_config.get_all_values()
    try:
        yield
    finally:
        _skimpy_config.set_all_values(vals)


@contextmanager
def num_threads_scope(num):
    with config_scope():
        _skimpy_config.set_value("parallelize_parts", num)
        yield
