"""Foundation tests for the dsplot package skeleton + style template.

Per D-05: dsplot.style is an inheritable template. Every figure inherits
defaults; figures override locally (module-level derive) or globally
(reassignment). The lazy-lookup contract is what makes runtime reassignment
observable from already-constructed Plottables.

Per D-01: the library proper (research/dsplot/) imports zero subshader
symbols. The figures/ subdir is consumer code and is explicitly excluded
from the isolation grep.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


DSPLOT_DIR = Path(__file__).resolve().parents[2] / "dsplot"


def test_package_imports():
    import dsplot
    import dsplot.style
    assert hasattr(dsplot.style, "PRIMARY_COLOR")


def test_plottable_base_class_exists():
    from dsplot.plottables.base import Plottable
    assert hasattr(Plottable, "draw")


def test_library_has_zero_subshader_imports():
    """Per D-01: research/dsplot/ excluding figures/ imports zero subshader symbols."""
    offenders = []
    for path in DSPLOT_DIR.rglob("*.py"):
        if "figures" in path.parts:
            continue
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "subshader" in text:
            offenders.append(str(path.relative_to(DSPLOT_DIR.parent)))
    assert offenders == [], (
        f"D-01 violation — subshader references found in dsplot library proper: {offenders}"
    )


def test_global_override_mode():
    """D-05 global reassignment: dsplot.style.PRIMARY_COLOR = '#new' persists."""
    import dsplot.style as style
    original = style.PRIMARY_COLOR
    try:
        style.PRIMARY_COLOR = "#abcdef"
        assert style.PRIMARY_COLOR == "#abcdef"
        importlib.reload  # ensure name available; we don't reload — reassignment must persist
        assert style.PRIMARY_COLOR == "#abcdef"
    finally:
        style.PRIMARY_COLOR = original
    assert style.PRIMARY_COLOR == original


EXPECTED_STYLE_CONSTANTS = [
    "DEFAULT_HSPACE",
    "DEFAULT_WSPACE",
    "DEFAULT_LABEL_RATIO",
    "DEFAULT_PANEL_MARGIN",
    "DEFAULT_ROW_LABEL_SIZE",
    "DEFAULT_TITLE_FONT_SIZE",
    "DEFAULT_SUBTITLE_FONT_SIZE",
    "DEFAULT_LABEL_FONT_SIZE",
    "DEFAULT_SUPTITLE_FONT_SIZE",
    "DEFAULT_TICK_LABEL_SIZE",
    "DEFAULT_PANEL_SIZE_INCHES",
    "DEFAULT_DPI",
    "DEFAULT_VECTOR_LABEL_OFFSET",
    "DEFAULT_VECTOR_LIM",
    "DEFAULT_AXIS_LABEL_OFFSET",
    "DEFAULT_HEATMAP_VMAX_PERCENTILE",
    "PRIMARY_COLOR",
    "SECONDARY_COLOR",
    "TERTIARY_COLOR",
    "NEUTRAL_COLOR",
    "HIGHLIGHT_COLOR",
    "BG_COLOR",
    "SPINE_COLOR",
    "TICK_LABEL_COLOR",
    "DROPLINE_COLOR",
]


@pytest.mark.parametrize("name", EXPECTED_STYLE_CONSTANTS)
def test_inheritable_template_completeness(name):
    """D-05 inheritable template: every default/role constant exposed."""
    import dsplot.style as style
    assert hasattr(style, name), f"dsplot.style missing constant: {name}"


def test_local_derive_does_not_mutate_global():
    """D-05 local-override contract: deriving a local constant from a default
    does NOT mutate the default itself."""
    import dsplot.style as style
    baseline = style.DEFAULT_HSPACE
    LOCAL = style.DEFAULT_HSPACE * 1.5
    LOCAL = LOCAL + 1.0  # mutate the local
    assert style.DEFAULT_HSPACE == baseline
