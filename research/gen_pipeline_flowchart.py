"""Generate a draw.io flowchart mapping the SubShader pipeline.

A hardware / memory swim-lane: three columns — DISK, CPU, GPU — with a single
flow threaded from start-up (top) through the runtime loop (bottom). Each step
sits in the column where its work runs; the important buffers are drawn inline
in the column where the data lives. Cross-column arrows are the data transfers
(disk→cpu read, cpu↔gpu copies) and are highlighted; same-column arrows are
plain. A "next frame" loop-back closes the runtime cycle.

No timing numbers — this is the structural/data-flow view. Output is plain
draw.io XML (open in app.diagrams.net or the VS Code Draw.io extension).
"""
from __future__ import annotations

import os
from xml.sax.saxutils import escape

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "assets", "timing", "pipeline_flowchart.drawio")

# Optichrome v52 palette (copied as constants — no dsplot dependency).
BG, FG = "#1A1A1A", "#EEEEEE"
AUDIO, DSP, RENDER, HILITE, SPINE = (
    "#ffd27d", "#7b6fe1", "#ff5a1f", "#22d3ee", "#444444")
WHITE_TXT = "#FFFFFF"

# Hardware lanes (left → right = the direction data physically moves).
LANES = ["disk", "cpu", "gpu"]
LANE_TITLE = {"disk": "DISK", "cpu": "CPU", "gpu": "GPU"}
LANE_C = {"disk": "#8c8c8c", "cpu": DSP, "gpu": RENDER}   # reuse plot palette

# --- geometry ---------------------------------------------------------------
LANE_W = 220
LANE_X = {"disk": 60, "cpu": 320, "gpu": 580}
BOX_W, BOX_H = 188, 52
BUF_W, BUF_H = 184, 56
TOP = 170
ROW_H = 64
RUNTIME_START = 13          # index in FLOW where the per-frame loop begins
PHASE_GAP = 50              # extra space for the START UP / RUNTIME divider

# Ordered pipeline. ('step'|'buf', id, label, lane). The flow threads through
# every node in order; buffers ('buf') are the memory the data lives in, drawn
# as cylinders in their lane. '· persists' marks a buffer built once and reused.
FLOW = [
    ("buf",  "audio_file",   "Audio File (WAV) · persists",   "disk"),
    ("step", "open_audio",   "Open Audio File",               "cpu"),
    ("step", "audio_out",    "Audio Output Init",             "cpu"),
    ("step", "build_kernels", "Build Wavelet Kernels",        "cpu"),
    ("step", "gen_fft",      "Generate FFT Kernel Bank",      "cpu"),
    ("buf",  "kbank_cpu",    "FFT Kernel Bank",               "cpu"),
    ("buf",  "kbank_gpu",    "Kernel Bank · persists",        "gpu"),
    ("step", "alloc_fb",     "Allocate Frame Buffer",         "cpu"),
    ("buf",  "frame_buffer", "Circular Frame Buffer · persists", "cpu"),
    ("step", "glctx",        "Create Window + GL Context",    "gpu"),
    ("step", "shader",       "Compile Shader + Texture",      "gpu"),
    ("buf",  "gl_texture",   "Frame Texture · persists",      "gpu"),
    ("step", "colormap",     "Color Map Init — pre-scan",     "gpu"),
    # ---- runtime loop (RUNTIME_START) ----
    ("step", "fetch",        "Fetch Audio Samples",           "cpu"),
    ("step", "fft_in",       "FFT Input Signal",              "cpu"),
    ("buf",  "spectrum_gpu", "Input Spectrum",                "gpu"),
    ("step", "multiply",     "Freq-Domain Multiply",          "gpu"),
    ("step", "ifft",         "IFFT Processed Data",           "gpu"),
    ("buf",  "coefs_cpu",    "CWT Coefficients",              "cpu"),
    ("step", "magnitude",    "Compute Magnitude",             "cpu"),
    ("step", "edges",        "Discard Edges",                 "cpu"),
    ("step", "hop",          "Advance Audio Hop",             "cpu"),
    ("step", "downsample",   "Downsample",                    "cpu"),
    ("step", "store",        "Store → Frame Buffer",          "cpu"),
    ("step", "tex_upload",   "Upload → Frame Texture",        "gpu"),
    ("step", "clear",        "Clear Previous",                "gpu"),
    ("step", "draw",         "Shader Draw",                   "gpu"),
    ("step", "swap",         "Update Display Buffer",         "gpu"),
    ("buf",  "display",      "Display Framebuffer",           "gpu"),
]


def _yrow(i):
    return TOP + i * ROW_H + (PHASE_GAP if i >= RUNTIME_START else 0)


class XML:
    def __init__(self):
        self.cells = []
        self.n = 0

    def _id(self, hint):
        self.n += 1
        return f"{hint}{self.n}"

    def text(self, x, y, w, h, value, *, size=20, color=FG, bold=True,
             align="left"):
        cid = self._id("t")
        style = (f"text;html=1;fontColor={color};fontSize={size};"
                 f"fontStyle={1 if bold else 0};align={align};"
                 f"verticalAlign=middle;")
        self.cells.append(
            f'<mxCell id="{cid}" value="{escape(value)}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" '
            f'height="{h}" as="geometry"/></mxCell>')
        return cid

    def lane(self, lane, top, height):
        """Full-height column background with a header chip."""
        color = LANE_C[lane]
        lx = LANE_X[lane]
        bg = self._id("lane")
        bg_style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor=#1d1d1d;"
                    f"strokeColor={color};strokeWidth=2;dashed=0;arcSize=3;")
        self.cells.append(
            f'<mxCell id="{bg}" value="" style="{bg_style}" vertex="1" '
            f'parent="1"><mxGeometry x="{lx}" y="{top}" width="{LANE_W}" '
            f'height="{height}" as="geometry"/></mxCell>')
        hd = self._id("laneh")
        hd_style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={color};"
                    f"strokeColor={color};fontColor=#1A1A1A;fontSize=16;"
                    f"fontStyle=1;align=center;verticalAlign=middle;arcSize=12;")
        self.cells.append(
            f'<mxCell id="{hd}" value="{LANE_TITLE[lane]}" style="{hd_style}" '
            f'vertex="1" parent="1"><mxGeometry x="{lx + 35}" y="{top - 20}" '
            f'width="{LANE_W - 70}" height="40" as="geometry"/></mxCell>')

    def step(self, key, label, lane, y):
        c = LANE_C[lane]
        x = LANE_X[lane] + (LANE_W - BOX_W) // 2
        style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={c};"
                 f"strokeColor={c};fontColor={WHITE_TXT};fontSize=12;"
                 f"fontStyle=1;align=center;verticalAlign=middle;arcSize=16;")
        self.cells.append(
            f'<mxCell id="{key}" value="{escape(label)}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{BOX_W}" '
            f'height="{BOX_H}" as="geometry"/></mxCell>')
        return key

    def buffer(self, key, label, lane, y):
        """A memory buffer — drawn as a cylinder so it reads as storage."""
        c = LANE_C[lane]
        x = LANE_X[lane] + (LANE_W - BUF_W) // 2
        style = (f"shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;"
                 f"fillColor=#202020;strokeColor={c};fontColor={c};fontSize=11;"
                 f"fontStyle=2;align=center;verticalAlign=middle;size=8;")
        self.cells.append(
            f'<mxCell id="{key}" value="{escape(label)}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{BUF_W}" '
            f'height="{BUF_H}" as="geometry"/></mxCell>')
        return key

    def edge(self, src, dst, *, color="#888888", label="", dashed=False,
             width=1.6, exit_xy=None, entry_xy=None, points=None):
        cid = self._id("e")
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
                 f"strokeColor={color};strokeWidth={width};endArrow=block;"
                 f"endFill=1;fontColor={color};fontSize=11;"
                 f"dashed={1 if dashed else 0};")
        if exit_xy:
            style += f"exitX={exit_xy[0]};exitY={exit_xy[1]};exitDx=0;exitDy=0;"
        if entry_xy:
            style += f"entryX={entry_xy[0]};entryY={entry_xy[1]};entryDx=0;entryDy=0;"
        pts = ""
        if points:
            inner = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points)
            pts = f'<Array as="points">{inner}</Array>'
        self.cells.append(
            f'<mxCell id="{cid}" value="{escape(label)}" style="{style}" '
            f'edge="1" parent="1" source="{src}" target="{dst}">'
            f'<mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>')
        return cid

    def render(self):
        body = "\n".join(self.cells)
        return (
            '<mxfile host="app.diagrams.net">\n'
            '  <diagram name="SubShader Pipeline">\n'
            '    <mxGraphModel dx="1200" dy="900" grid="0" gridSize="10" '
            'guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
            'page="1" pageScale="1" pageWidth="880" pageHeight="2160" '
            f'math="0" shadow="0" background="{BG}">\n'
            '      <root>\n'
            '        <mxCell id="0"/>\n'
            '        <mxCell id="1" parent="0"/>\n'
            f'{body}\n'
            '      </root>\n'
            '    </mxGraphModel>\n'
            '  </diagram>\n'
            '</mxfile>\n')


def build():
    x = XML()
    x.text(40, 22, 800, 34, "SubShader — Pipeline Data Flow (Disk · CPU · GPU)",
           size=22)

    bottom = _yrow(len(FLOW) - 1) + BUF_H + 24
    lane_top = TOP - 36
    for lane in LANES:
        x.lane(lane, lane_top, bottom - lane_top)

    # phase labels: START UP up top, RUNTIME divider before the loop section
    x.text(LANE_X["disk"], TOP - 64, 360, 22, "START UP  ·  one-time build",
           size=13, color="#BBBBBB")
    div_y = _yrow(RUNTIME_START) - PHASE_GAP // 2 - 6
    dl = x._id("div")
    x.cells.append(
        f'<mxCell id="{dl}" value="" style="rounded=0;fillColor={HILITE};'
        f'strokeColor={HILITE};" vertex="1" parent="1"><mxGeometry '
        f'x="{LANE_X["disk"]}" y="{div_y}" width="{LANE_X["gpu"] + LANE_W - LANE_X["disk"]}" '
        f'height="2" as="geometry"/></mxCell>')
    x.text(LANE_X["disk"], div_y - 24, 420, 22,
           "RUNTIME LOOP  ·  per audio frame", size=13, color=HILITE)

    # draw all nodes
    ids, lanes_by_id = [], {}
    for i, (kind, key, label, lane) in enumerate(FLOW):
        y = _yrow(i)
        if kind == "buf":
            x.buffer(key, label, lane, y)
        else:
            x.step(key, label, lane, y)
        ids.append(key)
        lanes_by_id[key] = lane

    # thread the flow: consecutive nodes; cross-lane hops are highlighted
    # transfers, same-lane links are plain.
    for a, b in zip(ids, ids[1:]):
        transfer = lanes_by_id[a] != lanes_by_id[b]
        x.edge(a, b, color=(HILITE if transfer else "#888888"),
               width=(2.0 if transfer else 1.6))

    # next-frame loop-back, routed up the clear left margin
    disp_y = _yrow(len(FLOW) - 1) + BUF_H // 2
    fetch_y = _yrow(RUNTIME_START) + BOX_H // 2
    x.edge("display", "fetch", color=HILITE, label="next frame", dashed=True,
           width=2.0, exit_xy=(0, 0.5), entry_xy=(0, 0.5),
           points=[(30, disp_y), (30, fetch_y)])
    return x


def main():
    x = build()
    with open(OUT, "w") as f:
        f.write(x.render())
    print(OUT)


if __name__ == "__main__":
    main()
