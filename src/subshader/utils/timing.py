"""Lightweight method timing decorator for pipeline profiling."""
import time
import functools


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
