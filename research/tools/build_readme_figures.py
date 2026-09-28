"""Regenerate every README figure from the shared style in one go.

usage: python research/tools/build_readme_figures.py <tag> [--skip fig1,drawio,timing,timing_extra,modules]

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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.dsplot import style
from research.dsplot.export import fit_readme_width

PY = sys.executable
DRAWIO = REPO / "assets/subshader.drawio"
PRIM_XML = REPO / "assets/subshader-primitives.xml"
SPLIT = REPO / ".planning/mocks/subshader_split.drawio"
TMP = REPO / ".planning/mocks/subshader_prim_tmp.drawio"
MODULES = REPO / ".planning/mocks/subshader_modules_gen.drawio"
SHOWN_PX = {"subshader_modules": 500}           # README width attribute where it isn't 100%
FRAMED = ("fig_1_", "subshader_modules", "timing_rate_check", "vertical_")   # the figures the README shows
FIG1_DIR = REPO / "assets/images/dsp/figures/by_figure/fig_1_fourier_vs_wavelet"


def run(*args):
    print("+", " ".join(str(a) for a in args))
    subprocess.run([str(a) for a in args], check=True, cwd=REPO)


def build_drawio(tag):
    run(PY, "research/tools/drawio_primitives.py", DRAWIO, TMP, PRIM_XML)
    run(PY, "research/tools/drawio_pipeline.py", TMP, DRAWIO, SPLIT)
    outs = []
    for page, name in (("Start Up Vertical", "startup"), ("Runtime Vertical", "runtime")):
        out = REPO / f"assets/images/drawio/vertical_{name}_{tag}_black.png"
        run(PY, "research/tools/drawio_export.py", SPLIT, page, out)
        outs.append(out)
    return outs


def build_modules(tag):
    out = REPO / f"assets/timing/subshader_modules_{tag}.png"
    run(PY, "research/tools/drawio_modules.py", MODULES)
    run(PY, "research/tools/drawio_export.py", MODULES, "0", out, "4.5")
    return [out]


def build_timing(tag):
    from research.dsplot.figures.gen_timing_hybrid import render_all
    return [Path(p) for p in render_all(f"timing_runtime_hybrid_{tag}.png", f"timing_startup_hybrid_{tag}.png",
                                        f"timing_runtime_deadline_{tag}.png", f"timing_rate_check_{tag}.png")]


def build_timing_extra(tag):
    sys.path.insert(0, str(REPO / "research"))
    import timing_report
    return [Path(timing_report.render_methods_figure(out=str(REPO / f"assets/timing/timing_methods_{tag}.png"))),
            Path(timing_report.render_config_figure(out=str(REPO / f"assets/timing/timing_config_{tag}.png")))]


def build_fig1(tag):
    from research.dsplot.figures.gen_figure_1_stft_vs_cwt import render_hero_split
    return [Path(p) for p in render_hero_split(str(FIG1_DIR), f"fig_1_fourier_vs_wavelet_hero_{tag}")]


def shown_px(p):
    return next((w for k, w in SHOWN_PX.items() if p.name.startswith(k)), style.README_SHOWN_PX)


def frame_px(a):
    """Shown thickness of the outer frame: the first bright run in from the left edge, mid-height."""
    row = a[a.shape[0] // 2].max(axis=1)
    x = int(np.argmax(row > 128))
    run = int(np.argmax(row[x:] <= 128))
    return run / style.README_SUPERSAMPLE


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
        palette = [np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) for c in trio | {"#FBFEA3"}]   # fig 1 heatmap ramp is exempt
        stray = {c for c in accents if min(np.abs(np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) - q).max() for q in palette) > 8}
        frame = frame_px(a) if p.name.startswith(FRAMED) else None
        good = (all(abs(m - style.README_EDGE_PAD_PX) <= 1 for m in margins) and tuple(a[0, 0]) == tuple(bg)
                and not (stray and p.name.startswith(("vertical", "timing", "subshader")))
                and w == shown_px(p) * style.README_SUPERSAMPLE
                and (frame is None or abs(frame - style.README_FRAME_PX) < 0.7))
        ok &= good
        print(f"{'ok ' if good else 'BAD'} {p.relative_to(REPO)}  {w}x{h} margins={tuple(int(m) for m in margins)} "
              f"frame={frame if frame is None else round(frame, 1)} accents={sorted(accents)}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--skip", default="")
    ns = ap.parse_args()
    skip = set(ns.skip.split(",")) - {""}
    outs = []
    for name, fn in (("drawio", build_drawio), ("modules", build_modules), ("timing", build_timing), ("timing_extra", build_timing_extra), ("fig1", build_fig1)):
        if name not in skip:
            outs += fn(ns.tag)
    for p in outs:
        fit_readme_width(str(p), shown_px(p))
    print("\n== check ==")
    sys.exit(0 if check(outs) else 1)
