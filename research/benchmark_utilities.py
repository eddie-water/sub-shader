"""
Benchmark utilities for live progress display and timing measurement.
"""

import sys

import numpy as np

# =============================================================================
# LIVE TABLE DISPLAY
# =============================================================================
BAR_WIDTH = 24
TABLE_WIDTH = 80


def _mini_bar(current: int, total: int) -> str:
    filled = int(BAR_WIDTH * current / total)
    return "\u2588" * filled + "\u2591" * (BAR_WIDTH - filled)


def _format_row(label: str, iteration: int, total: int, times_so_far) -> str:
    n = iteration
    if n > 0:
        avg = float(np.mean(times_so_far[:n]))
        mn  = float(np.min(times_so_far[:n]))
        mx  = float(np.max(times_so_far[:n]))
    else:
        avg = mn = mx = 0.0
    return (
        f"{label:<12}   "
        f"{_mini_bar(iteration, total)}   "
        f"{iteration:>5}/{total:<4}   "
        f"{avg:>10.2f}   "
        f"{mn:>10.2f}   "
        f"{mx:>10.2f}"
    )


def live_row(label: str, iteration: int, total: int, times_so_far):
    """Overwrite the current terminal line with a live table row."""
    sys.stdout.write(f"\r{_format_row(label, iteration, total, times_so_far)}")
    if iteration >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def live_progress(iteration: int, total: int):
    """Single-line progress update using \\r. Print results table after loop."""
    bar = _mini_bar(iteration, total)
    sys.stdout.write(f"\r{bar}    {iteration:>5}/{total}")
    sys.stdout.flush()


def print_figure_header(title: str):
    """Print the title block before processing begins.

    ================================================================================
    Chirp Signal Comparison
    --------------------------------------------------------------------------------
    """
    sep_heavy = "=" * TABLE_WIDTH
    sep_light = "-" * TABLE_WIDTH
    print(sep_heavy)
    print(title)
    print(sep_light)


def print_figure_results(num_frames: int, labels: list,
                         times_list: list, total_s: float, save_path: str):
    """Print the results block after processing completes.

    --------------------------------------------------------------------------------
    Function          Avg (ms)     Min (ms)     Max (ms)
    --------------------------------------------------------------------------------
    STFT                  1.44         1.29         4.45
    PyWavelet CWT       994.02       944.49      1189.25
    SubShader CWT        79.45        63.53       101.64
    --------------------------------------------------------------------------------
    Total Time: XXXX ms (Y s)
    --------------------------------------------------------------------------------
    Saved → path/to/file.png
    ================================================================================
    """
    sep_heavy = "=" * TABLE_WIDTH
    sep_light = "-" * TABLE_WIDTH

    print(sep_light)
    print(f"{'Function':<18}{'Avg (ms)':>12}{'Min (ms)':>12}{'Max (ms)':>12}")
    print(sep_light)
    for label, times in zip(labels, times_list):
        avg = float(np.mean(times))
        mn  = float(np.min(times))
        mx  = float(np.max(times))
        print(f"{label:<18}{avg:>12.2f}{mn:>12.2f}{mx:>12.2f}")
    print(sep_light)
    print(f"Total Time: {total_s * 1000:.2f} ms ({total_s:.2f} s)")
    print(sep_light)
    print(f"Saved \u2192 {save_path}")
    print(sep_heavy)


# -- Legacy helpers used by TimedSubShader --

TABLE_HEADER = f"{'Function':<12} {'Progress':<{BAR_WIDTH}} {'Iterations':>10} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}"
TABLE_SEP    = "-" * len(TABLE_HEADER)


def print_results_table(labels: list, times_list: list):
    """Print final results table after benchmarking completes."""
    print_header()
    for label, times in zip(labels, times_list):
        n = len(times)
        avg = float(np.mean(times))
        mn  = float(np.min(times))
        mx  = float(np.max(times))
        print(
            f"{label:<16}"
            f"{avg:>10.2f}   "
            f"{mn:>10.2f}   "
            f"{mx:>10.2f}   "
            f"({n} iterations)"
        )
    print_separator()


def print_separator():
    print(TABLE_SEP)


def print_header():
    print(TABLE_HEADER)
    print(TABLE_SEP)


def print_total(total_s: float):
    print(TABLE_SEP)
    print(f"{'Total Test Time':<32}{'':>{BAR_WIDTH + 16}}{total_s * 1000:>10.2f} ms  {total_s:>8.2f} s")


# =============================================================================
# TIMING
# =============================================================================


def compute_timing_stats(times: np.ndarray) -> dict:
    """Compute avg/min/max ms from a timing array."""
    return {
        "avg_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }
