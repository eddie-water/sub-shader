"""SubShader test suite — research, benchmarking, and figure generation.

Usage:
    python research/test_suite.py --timing          # Profile pipeline stages
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
    group.add_argument("--timing", action="store_true",
                       help="Profile pipeline stages; write results to assets/timing/")
    group.add_argument("--test", action="store_true",
                       help="Run pytest suite (research/tests/)")
    group.add_argument("--compare-methods", action="store_true",
                       help="Generate per-signal method comparison figures")
    group.add_argument("--figures", action="store_true",
                       help="Generate all documentation figures")

    parser.add_argument("--signal", type=str, default=None,
                        help="Signal name for --compare-methods (default: all signals)")
    parser.add_argument("--stub", action="store_true",
                        help="Stub PyWavelet calls with random data (faster)")
    parser.add_argument("--input-signal", type=str, default=None,
                        help="Custom audio file path for --compare-methods")

    args = parser.parse_args()

    if args.timing:
        import matplotlib
        matplotlib.use("Agg")
        from research.timing import run_timing
        run_timing(stub=args.stub)

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
