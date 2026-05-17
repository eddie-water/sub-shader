"""``python -m research.dsplot.figures`` — regenerate all DSP.md figures.

Output convention:
  - Default mode (no ``--suffix``): overwrites the canonical filenames at
    ``assets/images/dsp/<name>.png``. This is the "drop-in replacement"
    mode the orchestrator uses post-merge.
  - ``--suffix _09_05`` (or any string): writes to
    ``assets/images/dsp/<stem>_09_05.png`` so the new render coexists
    with the original for visual diff. This is the executor mode used
    during plan 09-05 to keep originals available for review.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

# Headless rendering — no GUI window pops up on dispatcher invocation.
matplotlib.use("Agg")

# Ensure ``import dsplot`` works regardless of cwd: insert research/ on path.
_RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

_OUT_DIR = os.path.abspath(
    os.path.join(_RESEARCH_DIR, "..", "assets", "images", "dsp")
)


def _with_suffix(filename: str, suffix: str) -> str:
    if not suffix:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}{suffix}{ext}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.dsplot.figures")
    parser.add_argument(
        "--suffix",
        default="",
        help=(
            "Filename suffix appended before the extension (e.g. '_09_05'). "
            "Empty (default) overwrites canonical filenames."
        ),
    )
    parser.add_argument(
        "--out",
        default=_OUT_DIR,
        help="Output directory (default: assets/images/dsp/).",
    )
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    # Defer imports until after sys.path is wired so ``import dsplot`` succeeds.
    from . import (
        components_recombine,
        dot_product_geometry,
        projection_reconstruction,
        vector_basics,
        vector_projection_3d,
    )

    paths: list[str] = []
    paths.append(vector_basics.render(
        args.out, _with_suffix("vector_basics.png", args.suffix)
    ))
    paths.append(dot_product_geometry.render(
        args.out, _with_suffix("dot_product_geometry.png", args.suffix)
    ))
    paths.append(components_recombine.render(
        args.out,
        _with_suffix("components_recombine_either_order_v18.png", args.suffix),
    ))
    paths.append(components_recombine.render_vector_xy_reconstruction(
        args.out,
        _with_suffix("vector_xy_reconstruction.png", args.suffix),
    ))
    paths.append(projection_reconstruction.render(
        args.out,
        _with_suffix("projection_reconstruction_either_order_v9.png", args.suffix),
    ))
    paths.append(vector_projection_3d.render(
        args.out,
        _with_suffix("vector_projection_3d_v2_combo5_palette.png", args.suffix),
    ))

    # Task 3 will add: motivator (all 6 versions), alignment_diagnostic

    for p in paths:
        print(f"  Saved -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
