"""Regenerate every README figure from the shared style in one go.

usage: python tools/build_readme_figures.py <tag> [--skip fig1,drawio,timing,modules]

<tag> is the version suffix for this build (e.g. v7); existing PNGs are never
overwritten, so bump it every build. Order matters: the Primitives page is
copied from the installed drawio, so primitives run before the pipeline.
Finishes with a check that every output has the shared background, edge pad
and accent palette, and prints the README paths to swap in.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from research.dsplot import style

PY = sys.executable
DRAWIO = REPO / "assets/subshader.drawio"
PRIM_XML = REPO / "assets/subshader-primitives.xml"
SPLIT = REPO / ".planning/mocks/subshader_split.drawio"
TMP = REPO / ".planning/mocks/subshader_prim_tmp.drawio"
FIG1_DIR = REPO / "assets/images/dsp/figures/by_figure/fig_1_fourier_vs_wavelet"


def run(*args):
    print("+", " ".join(str(a) for a in args))
    subprocess.run([str(a) for a in args], check=True, cwd=REPO)


def build_drawio(tag):
    run(PY, "tools/drawio_primitives.py", DRAWIO, TMP, PRIM_XML)
    run(PY, "tools/drawio_pipeline.py", TMP, DRAWIO, SPLIT)
    outs = []
    for page, name in (("Start Up Vertical", "startup"), ("Runtime Vertical", "runtime")):
        out = REPO / f"assets/images/drawio/vertical_{name}_{tag}_black.png"
        run(PY, "tools/drawio_export.py", SPLIT, page, out)
        outs.append(out)
    return outs


def build_modules(tag):
    out = REPO / f"assets/timing/subshader_modules_{tag}.png"
    run(PY, "tools/drawio_export.py", "assets/timing/subshader_modules.drawio", "0", out, "4.5")
    return [out]


def build_timing(tag):
    from research.dsplot.figures.gen_timing_hybrid import render_all
    return [Path(p) for p in render_all(f"timing_runtime_hybrid_{tag}.png", f"timing_startup_hybrid_{tag}.png",
                                        f"timing_runtime_deadline_{tag}.png", f"timing_rate_check_{tag}.png")]


def build_fig1(tag):
    from research.dsplot.figures.gen_figure_1_stft_vs_cwt import render_hero_split
    return [Path(p) for p in render_hero_split(str(FIG1_DIR), f"fig_1_fourier_vs_wavelet_hero_{tag}")]


def check(paths):
    trio = {c.upper() for c in (style.AUDIO_COLOR, style.DSP_COLOR, style.RENDER_COLOR)}
    bg = np.array([int(style.BG_COLOR.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)])
    ok = True
    for p in paths:
        a = np.asarray(Image.open(p).convert("RGB")).astype(int)
        h, w, _ = a.shape
        ys, xs = np.where(np.abs(a - bg).max(axis=2) > 20)
        margins = (xs.min(), ys.min(), w - 1 - xs.max(), h - 1 - ys.max())
        sat = a.reshape(-1, 3)[(a.max(axis=2) - a.min(axis=2)).reshape(-1) > 60]
        cols, cnt = np.unique(sat, axis=0, return_counts=True)
        accents = {"#%02X%02X%02X" % tuple(c) for c, n in zip(cols, cnt) if n > 0.05 * len(sat)} if len(sat) else set()
        stray = accents - trio - {"#FBFEA3"}          # fig 1 heatmap ramp is exempt
        good = all(m == style.README_EDGE_PAD_PX for m in margins) and tuple(a[0, 0]) == tuple(bg) and not (stray and p.name.startswith(("vertical", "timing", "subshader")))
        ok &= good
        print(f"{'ok ' if good else 'BAD'} {p.relative_to(REPO)}  {w}x{h} margins={tuple(int(m) for m in margins)} accents={sorted(accents)}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--skip", default="")
    ns = ap.parse_args()
    skip = set(ns.skip.split(",")) - {""}
    outs = []
    for name, fn in (("drawio", build_drawio), ("modules", build_modules), ("timing", build_timing), ("fig1", build_fig1)):
        if name not in skip:
            outs += fn(ns.tag)
    print("\n== check ==")
    sys.exit(0 if check(outs) else 1)
