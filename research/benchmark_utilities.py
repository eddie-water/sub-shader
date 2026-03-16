"""
Benchmark utilities for live progress display and timing measurement.
"""

import sys

import numpy as np

# =============================================================================
# LIVE TABLE DISPLAY
# =============================================================================
NEW_LINE = "\n"
BAR_WIDTH = 12
TABLE_HEADER = f"{'Function':<32}   {'Progress':<{BAR_WIDTH}}   {'Iterations':>10}   {'Avg (ms)':>10}   {'Min (ms)':>10}   {'Max (ms)':>10}"
TABLE_SEP    = "-" * len(TABLE_HEADER)


def _mini_bar(current: int, total: int) -> str:
    filled = int(BAR_WIDTH * current / total)
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def _format_row(label: str, iteration: int, total: int, times_so_far) -> str:
    n = iteration
    if n > 0:
        avg = float(np.mean(times_so_far[:n]))
        mn  = float(np.min(times_so_far[:n]))
        mx  = float(np.max(times_so_far[:n]))
    else:
        avg = mn = mx = 0.0
    return (
        f"{label:<32}   "
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


def live_rows(labels: list, iteration: int, total: int,
              times_list: list, num_rows: int):
    """Overwrite multiple terminal lines with live table rows."""
    if iteration > 1:
        sys.stdout.write(f"\033[{num_rows}A")
    for i in range(num_rows):
        sys.stdout.write(f"\r{_format_row(labels[i], iteration, total, times_list[i])}\n")
    sys.stdout.flush()


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
