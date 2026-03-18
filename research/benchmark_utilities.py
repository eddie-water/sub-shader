"""
Benchmark utilities for live progress display and timing measurement.

Output format follows the template in benchmark-timing-template.txt:
structured sections with ====/---- separators, init timing tables,
progress bars, and results tables with avg/max/min columns.
"""

import sys

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

TABLE_WIDTH = 80
COL_LABEL = 28
COL_VAL = 16
BAR_WIDTH = 24


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _mini_bar(current: int, total: int) -> str:
    filled = int(BAR_WIDTH * current / total)
    return "\u2588" * filled + "\u2591" * (BAR_WIDTH - filled)


def _sep_heavy():
    return "=" * TABLE_WIDTH


def _sep_light():
    return "-" * TABLE_WIDTH


# =============================================================================
# HUMAN-READABLE TIME
# =============================================================================

def format_time_human(ms: float) -> str:
    """Convert milliseconds to human-readable: '2 min 30 s', '5 s 123 ms', '123 ms'."""
    if ms < 1000:
        return f"{ms:.0f} ms"
    total_s = ms / 1000
    mins = int(total_s // 60)
    secs = int(total_s % 60)
    remaining_ms = int(round(ms - (mins * 60000 + secs * 1000)))
    if mins > 0:
        return f"{mins} min {secs} s"
    return f"{secs} s {remaining_ms} ms"


# =============================================================================
# SECTION BOUNDARIES
# =============================================================================

def print_section_start(title: str):
    """Print ===== / title / ----- to open a section."""
    print(_sep_heavy())
    print(title)
    print(_sep_light())


def print_section_end():
    """Print ===== to close a section."""
    print(_sep_heavy())


def print_separator():
    """Print a light separator line."""
    print(_sep_light())


# =============================================================================
# INIT TIMING TABLE (single-value column)
# =============================================================================

def print_init_header():
    print(f"{'Function':<{COL_LABEL}}{'Time (ms)':>{COL_VAL}}")
    print(_sep_light())


def print_init_row(label: str, time_ms: float):
    print(f"{label:<{COL_LABEL}}{time_ms:>{COL_VAL - 3}.2f} ms")


def print_init_total(total_ms: float):
    print(_sep_light())
    print(f"{'Total Init Time':<{COL_LABEL}}{total_ms:>{COL_VAL - 3}.2f} ms")
    print(f"{'':<{COL_LABEL}}{format_time_human(total_ms):>{COL_VAL}}")


# =============================================================================
# LIVE PROGRESS
# =============================================================================

def live_progress(current: int, total: int, unit: str = "frames"):
    """Single-line progress bar using \\r."""
    bar = _mini_bar(current, total)
    sys.stdout.write(f"\rProgress: {bar}  {current:>5} / {total} {unit}")
    sys.stdout.flush()


def clear_progress():
    """Clear the live progress line."""
    sys.stdout.write('\r' + ' ' * TABLE_WIDTH + '\r')
    sys.stdout.flush()


# =============================================================================
# RESULTS TABLE (three-value columns: avg / max / min)
# =============================================================================

def print_results_header():
    """Print the results column header with surrounding separators."""
    print(_sep_light())
    print(f"{'Function':<{COL_LABEL}}{'Avg (ms)':>{COL_VAL}}{'Max (ms)':>{COL_VAL}}{'Min (ms)':>{COL_VAL}}")
    print(_sep_light())


def print_results_row(label: str, times: np.ndarray):
    """Print one results row with avg/max/min computed from a timing array."""
    avg = float(np.mean(times))
    mx = float(np.max(times))
    mn = float(np.min(times))
    print(f"{label:<{COL_LABEL}}{avg:>{COL_VAL - 3}.2f} ms{mx:>{COL_VAL - 3}.2f} ms{mn:>{COL_VAL - 3}.2f} ms")


def print_loop_summary(n_loops: int, total_times: np.ndarray):
    """Print Total Loops count + Total Loop Time (avg/max/min) with human times."""
    avg = float(np.mean(total_times))
    mx = float(np.max(total_times))
    mn = float(np.min(total_times))
    print(_sep_light())
    print(f"{'Total Loops':<{COL_LABEL}}{n_loops:>{COL_VAL}}")
    print(_sep_light())
    print(f"{'Total Loop Time':<{COL_LABEL}}{avg:>{COL_VAL - 3}.2f} ms{mx:>{COL_VAL - 3}.2f} ms{mn:>{COL_VAL - 3}.2f} ms")
    print(f"{'':<{COL_LABEL}}{format_time_human(avg):>{COL_VAL}}{format_time_human(mx):>{COL_VAL}}{format_time_human(mn):>{COL_VAL}}")


def print_total_time(total_ms: float):
    """Print a single total time with human-readable line below."""
    print(_sep_light())
    print(f"{'Total Time':<{COL_LABEL}}{total_ms:>{COL_VAL - 3}.2f} ms")
    print(f"{'':<{COL_LABEL}}{format_time_human(total_ms):>{COL_VAL}}")


# =============================================================================
# TIMING STATS
# =============================================================================

def compute_timing_stats(times: np.ndarray) -> dict:
    """Compute avg/min/max ms from a timing array."""
    return {
        "avg_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


# =============================================================================
# MODE RUNNER
# =============================================================================

def run_modes(modes):
    """Run a list of (name, fn) modes with top-level step tracking when > 1."""
    n = len(modes)
    show_steps = n > 1
    for i, (name, fn) in enumerate(modes, 1):
        if show_steps:
            print(f"\n{'='*60}")
            print(f"  Step {i}/{n}: {name}")
            print(f"{'='*60}")
        fn()
    if show_steps:
        print(f"\n{'='*60}")
        print(f"  All {n} steps complete.")
        print(f"{'='*60}\n")
