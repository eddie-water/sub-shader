"""Lightweight method timing decorator for pipeline profiling."""
import time
import functools
from contextlib import contextmanager


@contextmanager
def timed_block(obj, name):
    """Record the elapsed milliseconds of a code block onto `obj`.

    Accumulates into the dict ``obj._init_substages`` keyed by ``name`` (created
    on first use). Lets a single constructor report a named breakdown of its
    sub-steps — the init analogue of the per-method @timed attribute.

    Usage:
        with timed_block(self, "build_kernels"):
            ...heavy work...
        # self._init_substages["build_kernels"] == elapsed_ms
    """
    store = getattr(obj, "_init_substages", None)
    if store is None:
        store = {}
        setattr(obj, "_init_substages", store)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[name] = store.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0


def timed(method):
    """Decorator that stores elapsed milliseconds as an instance attribute.

    After each call, sets self._timing_{method_name}_ms to the elapsed time.
    Overhead is ~1 microsecond per call — negligible for SubShader's 10-30 fps pipeline.

    Usage:
        @timed
        def some_method(self, data):
            ...

        obj.some_method(data)
        print(obj._timing_some_method_ms)  # elapsed in ms
    """
    attr = f"_timing_{method.__name__}_ms"

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = method(self, *args, **kwargs)
        setattr(self, attr, (time.perf_counter() - t0) * 1000.0)
        return result

    return wrapper
