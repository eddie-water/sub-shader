"""Canonical foundation-figure vectors — single source of truth for §2.4 figures.

DSP.md §2.4 mirrors these values in alt-text and LaTeX blocks. If a value
here changes, DSP.md must mirror the change.

Migrated from ``research/dsp_figures.py`` (where the values were inlined per
figure) — consolidating them here keeps the components-recombine,
projection-reconstruction, and 3D foundation figures consistent.
"""

# 2D vector ``a`` — primary subject of §2.4 components/recombine + projection
# figures.
A = (2.0, 3.0)

# 2D vector ``a'`` (a-prime) — same magnitude as ``a`` reflected across the
# y-axis. Used in the "perpendicular components" panel to show that flipping
# the x-component sign doesn't change the y-component.
A_PRIME = (-2.0, 3.0)

# 2D vector ``b`` — second vector for projection-reconstruction panels.
B = (3.0, 2.0)

# z-component for the 3D extension of ``a`` — used by vector_projection_3d.
A_Z = 1.5

# Square axis limit (±FOUND_LIM on x, y, z) for all §2.4 foundation figures.
# Set generously so neither A nor B run into the spine arrows.
FOUND_LIM = 6
