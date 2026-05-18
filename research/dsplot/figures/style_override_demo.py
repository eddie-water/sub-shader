"""dsplot style override demo — D-05 BOTH modes, side-by-side.

Run as:

    python -m research.dsplot.figures.style_override_demo

Produces three PNGs in ``assets/images/dsp/style_override_demo/``:

  * ``default_palette.png``   — baseline, uses the inherited dsplot.style template.
  * ``global_override.png``   — Mode 1 (D-05): reassigns ``dsplot.style.PRIMARY_COLOR``
                                 etc. Affects every figure rendered after.
  * ``local_override.png``    — Mode 2 (D-05): a single figure module defines its
                                 own module-local constants (``MY_PRIMARY`` etc.)
                                 and passes them to the renderer. ``dsplot.style.*``
                                 is NOT touched, so other figures keep the inherited
                                 default.

After the demo runs ``main()`` asserts that ``dsplot.style.PRIMARY_COLOR`` equals
its original default (proves the global reassignment was reversed in finally
AND the local override never touched the global).

Why both modes matter:
  * Mode 1 is the right tool when you want a project-wide palette change (e.g.
    swap dark-mode for light-mode, or apply a brand palette globally).
  * Mode 2 is the right tool when ONE figure needs to diverge from the template
    without rippling the change to siblings (e.g. ``motivator.py``'s
    ``LAYOUT_HSPACE = style.DEFAULT_HSPACE`` is a local override that DOES
    derive from the default; if you reassigned the module-level constant it
    would only affect motivator).

This demo is consumer code, not library code — it lives at
``research/dsplot/figures/`` alongside the canonical DSP.md figure renderers.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

# Ensure ``import dsplot`` works regardless of cwd: insert research/ on path.
# Mirrors research/dsplot/figures/__main__.py's bootstrap so this demo can be
# invoked as ``python -m research.dsplot.figures.style_override_demo`` from
# the repo root (the canonical invocation pattern).
_RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

import dsplot  # noqa: E402  — must come after sys.path bootstrap
from dsplot import Figure, StaticPanel, Vector, style  # noqa: E402


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# research/dsplot/figures/ -> repo root is three levels up
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
OUT_DIR = os.path.join(_REPO_ROOT, "assets", "images", "dsp", "style_override_demo")


def _render_one(filename: str,
                primary_color: str,
                secondary_color: str,
                tertiary_color: str) -> str:
    """Render a 1x3 figure with three vectors in the supplied colors.

    Colors are taken from the function arguments, NOT from ``dsplot.style`` —
    this is what lets the caller demonstrate the override modes by deciding
    where the colors come from at the call site.
    """
    fig = Figure(
        n_rows=1, n_cols=3,
        figsize=(style.DEFAULT_PANEL_SIZE_INCHES * 3,
                 style.DEFAULT_PANEL_SIZE_INCHES * 1.1),
        suptitle="dsplot style override demo",
    )
    for col, (color, label) in enumerate(
        ((primary_color,   "primary"),
         (secondary_color, "secondary"),
         (tertiary_color,  "tertiary")),
    ):
        panel = StaticPanel(
            title=label,
            lim=2.0,
            axis_style="arrow",
            axis_labels=True,
            show_border=False,
        )
        panel.add(Vector((1.5, 1.0),
                         color=color,
                         label=label[0],
                         linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH))
        fig.add_panel(panel, row=0, col=col)
    fig.render()

    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path)
    fig.close()
    return out_path


def _render_global_default(filename: str) -> str:
    """Render using the CURRENT ``dsplot.style.*`` values.

    A normal figure module would do exactly this: read the template defaults at
    render time. We expose it as its own helper so the demo can call it before
    AND after reassigning ``dsplot.style.PRIMARY_COLOR`` and observe the
    difference.
    """
    return _render_one(
        filename,
        dsplot.style.PRIMARY_COLOR,
        dsplot.style.SECONDARY_COLOR,
        dsplot.style.TERTIARY_COLOR,
    )


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------ Baseline ------
    default_path = _render_global_default("default_palette.png")

    original_primary   = dsplot.style.PRIMARY_COLOR
    original_secondary = dsplot.style.SECONDARY_COLOR
    original_tertiary  = dsplot.style.TERTIARY_COLOR

    try:
        # ------ Mode 1: GLOBAL reassignment (per D-05) ------
        # Reassigning dsplot.style.* affects every figure rendered after.
        dsplot.style.PRIMARY_COLOR   = "#22c55e"  # green
        dsplot.style.SECONDARY_COLOR = "#ec4899"  # pink
        dsplot.style.TERTIARY_COLOR  = "#06b6d4"  # cyan
        global_path = _render_global_default("global_override.png")

        # Restore originals before demonstrating the local-override mode so the
        # local demo's "global is untouched" assertion is meaningful.
        dsplot.style.PRIMARY_COLOR   = original_primary
        dsplot.style.SECONDARY_COLOR = original_secondary
        dsplot.style.TERTIARY_COLOR  = original_tertiary

        # ------ Mode 2: LOCAL figure override (per D-05) ------
        # Module-local constants for THIS figure only. Other figures still
        # inherit the global dsplot.style.* defaults — they are NOT touched.
        # This is the same pattern motivator.py uses to derive its own
        # LAYOUT_HSPACE / LAYOUT_LABEL_RATIO / etc. from the defaults.
        MY_PRIMARY   = "#facc15"  # yellow
        MY_SECONDARY = "#a855f7"  # purple
        MY_TERTIARY  = "#f97316"  # orange
        local_path = _render_one(
            "local_override.png", MY_PRIMARY, MY_SECONDARY, MY_TERTIARY,
        )

        # Load-bearing assertion: the local-override block must NOT mutate the
        # global dsplot.style.* template. If this fails, mode 2 leaked into
        # mode 1 (a D-05 violation).
        assert dsplot.style.PRIMARY_COLOR == original_primary, (
            "Local override should NOT mutate dsplot.style.PRIMARY_COLOR "
            "(D-05 violation: local-mode leaked into global-mode)"
        )
        assert dsplot.style.SECONDARY_COLOR == original_secondary
        assert dsplot.style.TERTIARY_COLOR == original_tertiary
    finally:
        # Belt-and-braces: restore originals even if a raise interrupted the
        # local block. Test-pollution defense.
        dsplot.style.PRIMARY_COLOR   = original_primary
        dsplot.style.SECONDARY_COLOR = original_secondary
        dsplot.style.TERTIARY_COLOR  = original_tertiary

    print(f"  Saved -> {default_path}")
    print(f"  Saved -> {global_path}   (mode 1: global reassignment, affects all)")
    print(f"  Saved -> {local_path}    (mode 2: local-only, global untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
