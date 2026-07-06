"""SubShader test suite — research, benchmarking, and figure generation.

Usage:
    python research/test_suite.py --live-timing      # Profile the live pipeline on a song clip
    python research/test_suite.py --sweep            # Sweep configs + comparison plot
    python research/test_suite.py --test             # Run pytest suite
    python research/test_suite.py --compare-methods  # Per-signal method comparison figures
    python research/test_suite.py --figures          # Generate all documentation figures

Optional flags:
    --signal <name>         Limit --compare-methods to one signal (chirp, polyphonic, musical)
    --stub                  Stub PyWavelet calls with random data (faster iteration)
    --input-signal <path>   Custom audio file path for --compare-methods
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="SubShader test suite")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--live-timing", action="store_true",
                       help="Profile the live pipeline (audio+CWT+renderer) on a song clip")
    group.add_argument("--sweep", action="store_true",
                       help="Sweep backend/chunk_size/overlap; write CSV + comparison PNG")
    group.add_argument("--methods", action="store_true",
                       help="Time STFT/PyWavelet/CWT-CPU/CWT-GPU on real audio (no window)")
    group.add_argument("--test", action="store_true",
                       help="Run pytest suite (research/tests/)")
    group.add_argument("--compare-methods", action="store_true",
                       help="Generate per-signal method comparison figures")
    group.add_argument("--figures", action="store_true",
                       help="Generate all documentation figures")

    # Single-run config knobs (used by --live-timing; the sweep drives these per child)
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Override chunk_size for --live-timing")
    parser.add_argument("--overlap", type=float, default=None,
                        help="Override overlap_factor for --live-timing")
    parser.add_argument("--octaves", type=int, default=None,
                        help="Override num_octaves for --live-timing")
    parser.add_argument("--backend", type=str, default=None, choices=["gpu", "cpu"],
                        help="Force CWT backend for --live-timing")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Seconds of audio to profile for --live-timing")
    parser.add_argument("--no-pace", action="store_true",
                        help="Run frames back-to-back (raw compute) instead of real-time paced")
    parser.add_argument("--quiet", action="store_true",
                        help="Compact one-line output (used by the sweep)")

    parser.add_argument("--signal", type=str, default=None,
                        help="Signal name for --compare-methods (default: all signals)")
    parser.add_argument("--stub", action="store_true",
                        help="Stub PyWavelet calls with random data (faster)")
    parser.add_argument("--input-signal", type=str, default=None,
                        help="Custom audio file path for --compare-methods")

    args = parser.parse_args()

    if args.live_timing:
        import matplotlib
        matplotlib.use("Agg")
        from timing_live import run_live_timing
        kwargs = {}
        if args.chunk_size is not None:
            kwargs["chunk_size"] = args.chunk_size
        if args.overlap is not None:
            kwargs["overlap_factor"] = args.overlap
        if args.octaves is not None:
            kwargs["num_octaves"] = args.octaves
        if args.backend is not None:
            kwargs["backend"] = args.backend
        if args.seconds is not None:
            kwargs["seconds"] = args.seconds
        if args.no_pace:
            kwargs["paced"] = False
        if args.quiet:
            kwargs["quiet"] = True
            # Sweep mode: keep transform + render as single blobs for clean,
            # consistent config-comparison stacks (don't split into legs).
            kwargs["profile_gpu_transfers"] = False
            kwargs["profile_render"] = False
        run_id = run_live_timing(**kwargs)
        if run_id:
            print(f"RUN_ID={run_id}")

    elif args.sweep:
        import matplotlib
        matplotlib.use("Agg")
        from timing_sweep import run_sweep
        run_sweep()

    elif args.methods:
        import matplotlib
        matplotlib.use("Agg")
        from timing_methods import run_methods
        run_methods()

    elif args.test:
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "research/tests/", "-v"],
            cwd=project_root,
            check=False,
        )
        sys.exit(result.returncode)

    elif args.compare_methods:
        import matplotlib
        matplotlib.use("Agg")
        from research.figures import generate_method_comparison
        generate_method_comparison(
            signal_name=args.signal,
            input_signal=args.input_signal,
            stub=args.stub,
        )

    elif args.figures:
        import matplotlib
        matplotlib.use("Agg")
        from research.figures import generate_all_figures
        generate_all_figures(stub=args.stub)


if __name__ == "__main__":
    main()
