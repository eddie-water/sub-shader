"""DEPRECATED — DSP.md figure generators moved to ``research/dsplot/figures/``.

Backwards-compatibility shim. Importing this module emits a DeprecationWarning.

Migration map:
  A, A_PRIME, B, A_Z, FOUND_LIM  -> dsplot.figures.foundation_constants
  ChirpFigureConfig              -> dsplot.figures.motivator.MotivatorConfig
  MOTIVATOR_VERSIONS             -> dsplot.figures.motivator.VERSIONS
  render_motivator(cfg)          -> dsplot.figures.motivator.render_one
  generate_motivator_versions()  -> dsplot.figures.motivator.render_all
  generate_alignment_diagnostic  -> dsplot.figures.alignment_diagnostic.render
  generate_foundations_figures   -> dispatched by dsplot.figures.__main__
  generate_all_dsp_figures()     -> dsplot.figures.__main__.main

``python research/dsp_figures.py`` still works — delegates to the new dispatcher.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "research.dsp_figures is deprecated — use research.dsplot.figures instead. "
    "See the migration map in this module's docstring.",
    DeprecationWarning,
    stacklevel=2,
)

# Foundation constants (canonical vector values for §2.4 figures).
from research.dsplot.figures.foundation_constants import (  # noqa: E402,F401
    A,
    A_PRIME,
    A_Z,
    B,
    FOUND_LIM,
)

# Motivator: aliases mapping legacy names to the new ones.
from research.dsplot.figures.motivator import (  # noqa: E402
    MotivatorConfig as ChirpFigureConfig,
    VERSIONS as MOTIVATOR_VERSIONS,
    render_all as generate_motivator_versions,
    render_one as render_motivator,
)

# Alignment diagnostic: legacy name -> new render() entry point.
from research.dsplot.figures.alignment_diagnostic import (  # noqa: E402
    render as generate_alignment_diagnostic,
)


def generate_all_dsp_figures() -> int:
    """Regenerate every DSP.md figure via the dsplot dispatcher.

    Delegates to ``research.dsplot.figures.__main__.main`` so the legacy
    command ``python research/dsp_figures.py`` keeps producing the same set
    of PNGs at the same paths.
    """
    from research.dsplot.figures.__main__ import main
    return main([])


if __name__ == "__main__":
    raise SystemExit(generate_all_dsp_figures())
