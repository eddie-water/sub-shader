"""
SubShader Benchmark Suite.

Modes:
  (default)                   Run all modes (same as --all)
  --timing                    STFT vs PyWavelet vs NumPy CWT vs CuPy CWT timing comparison
  --timing-chart              Generate timing bar chart PNG (assets/images/benchmarks/timing_bar_chart.png)
  --figures                   Generate 3 README comparison PNGs (matplotlib)
  --figures --stub            Generate stub layouts instead of real DSP (fast iteration)
  --figures --stub-pywt       Skip PyWavelet, use random stubs, save to stub folder (faster)
  --seaborn                   Generate 3 comparison PNGs (seaborn heatmap style)
  --seaborn --stub-pywt       Seaborn with stubbed PyWavelet, save to stub folder
  --comparison-grid           Generate 3x3 comparison grid (signals x representations)
  --unit-tests                Run unit tests (NumPy vs CuPy verification, etc.)
  --all                       Run timing, figures (matplotlib), figures (seaborn), unit tests
"""

import argparse
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')

from figures import ReadmeFigures, generate_comparison_grid, generate_timing_bar_chart
from timing import TimedSubShader, run_default
from wav_export import export_signal_to_wav  # noqa: F401

from utilities import run_modes, gpu_available

GPU_AVAILABLE = gpu_available()
if not GPU_AVAILABLE:
    print("[benchmark] No GPU detected -- CuPy benchmarks will be skipped.\n")

try:
    import seaborn  # noqa: F401
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("[benchmark] seaborn not installed -- --seaborn flag will be ignored.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SubShader benchmark suite"
    )
    parser.add_argument("--timing",          action="store_true", help="STFT vs PyWavelet vs CWT timing comparison")
    parser.add_argument("--timing-chart",    action="store_true", help="Generate timing bar chart PNG (timing_bar_chart.png)")
    parser.add_argument("--figures",         action="store_true", help="Generate all 3 README comparison PNGs (matplotlib)")
    parser.add_argument("--figures-chirp",      action="store_true", help="Generate chirp signal comparison figure only")
    parser.add_argument("--figures-polyphonic",  action="store_true", help="Generate polyphonic signal comparison figure only")
    parser.add_argument("--figures-musical",     action="store_true", help="Generate musical signal comparison figure only")
    parser.add_argument("--seaborn",         action="store_true", help="Generate 3 comparison PNGs (seaborn heatmap style)")
    parser.add_argument("--comparison-grid", action="store_true", help="Generate 3x3 comparison grid (signals x representations)")
    parser.add_argument("--comparison", action="store_true", help="Run all methods (STFT, PyWavelet, SubShader) with timing stats and produce comparison grid")
    parser.add_argument("--unit-tests",      action="store_true", help="Run unit tests (NumPy vs CuPy, etc.)")
    parser.add_argument("--all",             action="store_true", help="Run all modes")
    parser.add_argument("--stub",      action="store_true", help="With --figures: generate stub layouts instead of real DSP (fast iteration)")
    parser.add_argument("--stub-pywt", action="store_true", help="With --figures: skip PyWavelet computation, use random stub spectrograms (faster, saves to stub folder)")
    parser.add_argument("--dpi",       type=int, default=0, help="Output DPI for --comparison-grid. When set, output is named comparison_grid_{dpi}dpi.png. Default: 200 DPI with standard filename.")
    args = parser.parse_args()

    any_figure_individual = args.figures_chirp or args.figures_polyphonic or args.figures_musical
    any_flag = (args.timing or args.figures or any_figure_individual or args.seaborn
                or args.unit_tests or args.stub or args.comparison_grid or args.comparison
                or args.timing_chart)
    if args.all or not any_flag:
        args.timing = args.figures = args.seaborn = args.unit_tests = True

    if args.seaborn and not SEABORN_AVAILABLE:
        print("[benchmark] --seaborn requested but seaborn is not installed. "
              "pip install seaborn\n")
        args.seaborn = False

    modes = []
    if args.timing:
        modes.append(("Timing Comparison", lambda: TimedSubShader().run()))

    if args.figures or any_figure_individual:
        backends = ["matplotlib"]
        if args.seaborn:
            backends.append("seaborn")
        if args.stub:
            modes.append(("Stub Layouts", lambda: ReadmeFigures().stub_layouts()))
        elif any_figure_individual and not args.figures:
            rf = ReadmeFigures(backends=backends, stub_pywt=args.stub_pywt)
            if args.figures_chirp:
                modes.append(("Chirp Figure", lambda r=rf: r.chirp_signal_comparison()))
            if args.figures_polyphonic:
                modes.append(("Polyphonic Figure", lambda r=rf: r.polyphonic_signal_comparison()))
            if args.figures_musical:
                modes.append(("Musical Figure", lambda r=rf: r.musical_signal_comparison()))
        else:
            modes.append(("Figures", lambda b=backends, sp=args.stub_pywt: ReadmeFigures(backends=b, stub_pywt=sp).run_all()))
    elif args.seaborn:
        modes.append(("Seaborn Figures", lambda sp=args.stub_pywt: ReadmeFigures(backends=["seaborn"], stub_pywt=sp).run_all()))

    if args.comparison:
        modes.append(("Comparison", lambda sp=args.stub_pywt, d=args.dpi: generate_comparison_grid(stub_pywt=sp, dpi=d, comparison=True)))

    if args.comparison_grid:
        modes.append(("Comparison Grid", lambda sp=args.stub_pywt, d=args.dpi: generate_comparison_grid(stub_pywt=sp, dpi=d)))

    if args.timing_chart:
        modes.append(("Timing Bar Chart", lambda d=args.dpi: generate_timing_bar_chart(dpi=d if d > 0 else 200)))

    if args.unit_tests:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "src/", "-v"],
            cwd=project_root,
        )
        if result.returncode != 0:
            sys.exit(result.returncode)
        if not modes:
            sys.exit(0)

    if modes:
        run_modes(modes)
    else:
        run_default()
