"""Compose the Startup and Runtime swimlane diagrams from drawio primitives.

Usage:
  python tools/drawio_pipeline.py <in.drawio> <out.drawio>

Rebuilds the "Startup" and "Runtime" pages; every other page is kept.

Layout (top to bottom), all on the 10 px grid with a 50 px design unit:
  HW lane (top)      black square boxes: disk, sound device
  CPU lane           asset row (outer: framed tile + caption), then the stage row tight against the rail
  ===== rail =====   CPU | GPU boundary; every stage letter sits in a box on the rail, transfers in bold boxes
  GPU lane           stage row tight against the rail, then the asset row (outer)
  HW lane (bottom)   black square boxes: display

Stages are 2 x 2 units, gutters 1 unit. Every arrow leaves and enters a box at the centre of a side.
A lane change is one vertical drop in the gutter, through the transfer's box on the rail.
"""
import re, sys
from xml.sax.saxutils import escape

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import drawio_primitives as prim

AUDIO, DSP, REND = "#E8A317", "#6A5CD6", "#F0521A"
INK, GRAY, LANE = "default", "#888888", "#999999"   # "default" follows the draw.io theme: black/white on light, inverted on dark

U = 40                          # design unit (draw.io grid stays 10 px)
UNIT, GUT = 2 * U, U            # stage / asset frame = 2 x 2 units, gutter = 1 unit
COL = UNIT + GUT
X0 = 180                        # first column's left gutter centre; boxes start at X0 + GUT/2 = 200
LABEL_X, LABEL_W = 40, 120      # lane titles
CPU_H, GPU_H = 350, 290
LOOP_TOP = 60                   # room above the lanes for the runtime loop rail
LETTER_W, LETTER_H = U, U       # letter box on the rail, one per column (stages and transfers alike)
FRAME_PAD = 10                  # inset of the content inside its UNIT cell (decks and rasters fill the cell)
STRIP = (20, 80)                # the colour map: a half-unit strip, one UNIT tall, centred in its cell
FONT = 15                       # stage labels, captions, HW labels


def snap(v, g=10):
    return int(round(v / g) * g)


def col_x(c):
    return X0 + c * COL


class Page:
    def __init__(self, name, n_cols):
        self.name, self.cells, self.late, self.n = name, [], [], 0
        self.width = X0 + n_cols * COL + 40

    def cid(self):
        self.n += 1
        return f"{self.name[:2].lower()}{self.n}"

    def vertex(self, style, x, y, w, h, value="", late=False):
        i = self.cid()
        (self.late if late else self.cells).append(f'''
        <mxCell id="{i}" value="{escape(value.replace(chr(10), "<br>"))}" style="{style}" vertex="1" parent="1">
          <mxGeometry x="{snap(x)}" y="{snap(y)}" width="{snap(w)}" height="{snap(h)}" as="geometry" />
        </mxCell>''')
        return i

    def edge(self, src, dst, color=INK, dashed=False, width=3, exit_=None, entry=None, points=(), arrow=True):
        i = self.cid()
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor={color};strokeWidth={width};curved=0;"
                 + ("endArrow=block;endFill=1;endSize=6;" if arrow else "endArrow=none;")
                 + ("dashed=1;" if dashed else ""))
        if exit_:
            style += f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
        if entry:
            style += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        pts = "".join(f'<mxPoint x="{snap(px)}" y="{snap(py)}" />' for px, py in points)
        arr = f'<Array as="points">{pts}</Array>' if pts else ""
        self.cells.append(f'''
        <mxCell id="{i}" value="" style="{style}" edge="1" parent="1" source="{src}" target="{dst}">
          <mxGeometry relative="1" as="geometry">{arr}</mxGeometry>
        </mxCell>''')
        return i

    def text(self, value, x, y, w, h, size=20, color=INK, bold=True, bg=False, late=False):
        style = (f"text;html=1;strokeColor=none;fillColor={'default' if bg else 'none'};align=center;verticalAlign=middle;whiteSpace=wrap;"
                 f"fontColor={color};fontSize={size};" + ("fontStyle=1;" if bold else "") + "labelBackgroundColor=none;")
        return self.vertex(style, x, y, w, h, value, late=late)

    def box(self, label, x, y, color, rounded=True, w=UNIT, h=UNIT, size=FONT, late=False, stroke=3, font=INK):
        style = (f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;fillColor=default;strokeColor={color};fontColor={font};"
                 f"fontSize={size};fontStyle=1;align=center;verticalAlign=middle;arcSize=16;strokeWidth={stroke};labelBackgroundColor=none;")
        return self.vertex(style, x, y, w, h, label, late=late)

    def tile(self, key, x, y, color, caption, caption_side, size=None, width=2):
        """A primitive fitted into a UNIT cell at (x, y), the same footprint as a stage box. Rasters (heatmaps) get a
        black frame; decks and the colour-map strip stand on their own. Caption cell above (CPU row) or below (GPU
        row). Returns (id arrows attach to, cell geometry)."""
        it = next(i for i in prim.items if i["key"] == key)
        inner = UNIT if key.startswith(("deck_", "heatmap", "cmap")) else UNIT - 2 * FRAME_PAD
        if size:
            w, h = size
        else:                                                   # fitted side snaps to 20 so the centred content stays on grid
            w, h = inner, max(20, snap(inner * it["h"] / it["w"], 20))
            if h > inner:
                h, w = inner, max(20, snap(inner * it["w"] / it["h"], 20))
        if it.get("fillcolor"):
            fill = f"fillColor={it['fillcolor']};"
        elif it["fill"] or it["filled"]:
            fill = f"fillColor={color};fillOpacity=25;"
        else:
            fill = "fillColor=none;"
        gx, gy = snap(x + (UNIT - w) / 2), snap(y + (UNIT - h) / 2)
        anchor = self.box("", x, y, INK, rounded=False, w=UNIT, h=UNIT, stroke=2) if key.startswith("heatmap") else None
        style = f"shape=stencil({prim.stencils[key]});strokeColor={color};strokeWidth={width};{fill}"
        vid = self.vertex(style, gx, gy, w, h)
        cy = y - UNIT if caption_side == "above" else y + UNIT           # a UNIT caption cell stacked on the tile
        self.text(caption, x, cy, UNIT, UNIT, size=FONT, late=True)
        return anchor or vid, (x, y, UNIT, UNIT)

    def lane(self, title, y, h):
        self.vertex(f"rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor={LANE};strokeWidth=1;", LABEL_X, y, self.width - LABEL_X, h)
        self.text(title, LABEL_X, y, LABEL_W, h, size=22)

    def rail(self, y):
        """Double line at y-20 and y (the 'line' shape draws through the centre of its box); letter boxes centre on y-10."""
        for dy in (-30, -10):
            self.vertex(f"line;strokeWidth=2;html=1;strokeColor={INK};fillColor=none;", LABEL_X, y + dy, self.width - LABEL_X, 20)

    def rail_box(self, letter, cx, y_div, bold=False):
        """Letter box centred on the double line at x = cx; bold (double weight) for a CPU <-> GPU transfer."""
        return self.box(letter, cx - LETTER_W / 2, y_div - 30, INK, rounded=False, w=LETTER_W, h=LETTER_H, size=20, late=True, stroke=4 if bold else 2)

    def xml(self, height):
        return f'''  <diagram name="{self.name}" id="pipe-{self.name.lower()}">
    <mxGraphModel dx="1400" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{int(self.width) + 40}" pageHeight="{int(height)}" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />{"".join(self.cells + self.late)}
      </root>
    </mxGraphModel>
  </diagram>'''


def build(name, stages, resources, hw, data=(), loop_back=False):
    """Two lanes, four rows: CPU assets | CPU stages | rail | GPU stages | GPU assets. Hardware sits inline in the asset rows.
       stages:    (col, lane, label, color, letter) with lane in CPU | GPU, or
                  (col, "XFER", None, color, letter)    transfer column: the flow drops through its bold letter box, or
                  (col, "BRANCH", None, color, letter)  transfer column that branches off the previous stage's flow line
                                                        straight down/up into the asset framed in the same column
       resources: (col, lane, key, caption, color[, (w, h)])
       hw:        (col, lane, label)                    black box in that lane's asset row, referenced as 'H:<col>'
       data:      extra arrows: (src, dst, exit, entry, points) with src/dst 'S:<letter>', 'R:<col>:<lane>' or 'H:<col>'
                  and points as (col, dx, dy) offsets from col_x(col), y_cpu"""
    n_cols = max([s[0] for s in stages] + [h[0] for h in hw] + [r[0] for r in resources]) + 1
    pg = Page(name, n_cols)

    y_cpu = LOOP_TOP if loop_back else 0
    y_div = y_cpu + CPU_H                 # rail (lower line = lane boundary)
    y_gpu = y_div
    total_h = y_gpu + GPU_H + 40

    pg.lane("CPU", y_cpu, CPU_H)
    pg.lane("GPU", y_gpu, GPU_H)
    pg.rail(y_div)

    # rows (offsets from the lane top), in units of U with half-unit gaps
    #   CPU: caption cell 0..80 | asset / hardware cell 80..160 | stage box 220..300 | letter box y_div-30..y_div+10
    #   GPU: stage box 30..110 | asset / hardware cell 130..210 | caption cell 210..290
    cpu_tile_y, cpu_box_y = y_cpu + 80, y_cpu + 220
    gpu_box_y, gpu_tile_y = y_gpu + 30, y_gpu + 130
    mid = {"CPU": cpu_box_y + UNIT / 2, "GPU": gpu_box_y + UNIT / 2}

    stage_ids, geom = {}, {}
    for col, lane, label, color, letter in stages:
        if lane not in ("CPU", "GPU"):
            continue
        x = col_x(col) + GUT / 2
        y = cpu_box_y if lane == "CPU" else gpu_box_y
        stage_ids[letter] = pg.box(label, x, y, color)
        geom[f"S:{letter}"] = (x, y, UNIT, UNIT)
        pg.rail_box(letter, x + UNIT / 2, y_div)

    tile_ids = {}
    for entry in resources:
        col, lane, key, caption, color = entry[:5]
        size = entry[5] if len(entry) > 5 else None
        x = col_x(col) + GUT / 2
        y = cpu_tile_y if lane == "CPU" else gpu_tile_y
        tid, g = pg.tile(key, x, y, color, caption, "above" if lane == "CPU" else "below", size=size)
        tile_ids[(col, lane)] = tid
        geom[f"R:{col}:{lane}"] = g

    hw_ids = {}
    for col, lane, label in hw:                                 # black boxes, inline with the assets of their lane
        x = col_x(col) + GUT / 2
        y = cpu_tile_y if lane == "CPU" else gpu_tile_y
        hw_ids[col] = pg.box(label, x, y, INK, rounded=False)
        geom[f"H:{col}"] = (x, y, UNIT, UNIT)

    # flow: straight between stages; a lane change drops through the transfer column's box (or, unboxed, the gutter)
    prev, pending = None, None
    for st in stages:
        col, lane, label, color, letter = st
        cx = col_x(col) + GUT / 2 + UNIT / 2
        if lane == "XFER":
            pending = st
            pg.rail_box(letter, cx, y_div, bold=True)
            continue
        if lane == "BRANCH":                                     # fork off the flow line, straight down/up into the asset below/above
            far = "GPU" if prev[1] == "CPU" else "CPU"
            pg.edge(stage_ids[prev[4]], tile_ids[(col, far)], exit_=(1, 0.5), entry=(0.5, 0) if far == "GPU" else (0.5, 1),
                    points=[(cx, mid[prev[1]])])
            pg.rail_box(letter, cx, y_div, bold=True)
            continue
        if prev is not None:
            a, b = stage_ids[prev[4]], stage_ids[letter]
            if prev[1] == lane:
                pg.edge(a, b, exit_=(1, 0.5), entry=(0, 0.5))
            else:
                gx = col_x(pending[0]) + GUT / 2 + UNIT / 2 if pending else col_x(col)
                pg.edge(a, b, exit_=(1, 0.5), entry=(0, 0.5), points=[(gx, mid[prev[1]]), (gx, mid[lane])])
        prev, pending = st, None

    def ref(r):
        if r.startswith("S:"):
            return stage_ids[r[2:]]
        if r.startswith("H:"):
            return hw_ids[int(r[2:])]
        return tile_ids[(int(r.split(":")[1]), r.split(":")[2])]

    for src, dst, exit_, entry, points in data:                 # access / allocate / store: flow style, dashed
        pts = [(col_x(c) + dx, y_cpu + dy) for c, dx, dy in points]
        pg.edge(ref(src), ref(dst), dashed=True, exit_=exit_, entry=entry, points=pts)

    if loop_back:
        real = [s for s in stages if s[1] in ("CPU", "GPU")]
        first, last = real[0], real[-1]
        y_rail = 30
        x_last = col_x(last[0]) + COL
        x_first = col_x(first[0])
        pg.edge(stage_ids[last[4]], stage_ids[first[4]], exit_=(1, 0.5), entry=(0, 0.5),
                points=[(x_last, mid[last[1]]), (x_last, y_rail), (x_first, y_rail), (x_first, mid[first[1]])])
        pg.text("loop: every frame", snap((x_first + x_last) / 2 - 130), y_rail - 30, 260, 30, size=16, color=GRAY, bold=False)

    return pg.xml(total_h)


STARTUP_STAGES = [                       # all host code: allocations peel off the flow line, nothing runs on the GPU in sequence
    (0, "CPU", "Open\nAudio File", AUDIO, "A"),
    (1, "CPU", "Audio\nOutput\nInit", AUDIO, "B"),
    (2, "CPU", "Init CUDA", DSP, "C"),
    (3, "CPU", "Build\nWavelet\nKernels", DSP, "D"),
    (4, "CPU", "FFT\nKernel Bank", DSP, "E"),
    (5, "BRANCH", None, DSP, "F"),
    (6, "CPU", "Allocate\nFrame\nBuffer", REND, "G"),
    (7, "CPU", "OpenGL\nContext", REND, "H"),
    (8, "CPU", "Compile\nShader", REND, "I"),          # GPURenderer: shader (colour map baked into fragment.glsl) ...
    (9, "CPU", "GL\nTexture", REND, "J"),              # ... then the float texture the ring is written into
]
STARTUP_RESOURCES = [
    (3, "CPU", "deck_wavelets", "Wavelet\nBank [T]", DSP),
    (5, "GPU", "deck_freq", "Wavelet\nBank [F]", DSP),
    (6, "CPU", "deck_checker", "Frame\nBuffer", REND),                # orange card outlines, black/white checkers
    (8, "GPU", "cmap_vertical", "Color Map", INK, STRIP),
    (9, "GPU", "heatmap_square_weave_gray", "GPU Texture", INK),       # fresh texture: weave test pattern in ink
]
STARTUP_HW = [(0, "CPU", "Audio\nFile\n(Disk)"), (1, "CPU", "Audio\nDevice")]
STARTUP_DATA = [
    ("H:0", "S:A", (0.5, 1), (0.5, 0), []),                                     # disk -> open
    ("H:1", "S:B", (0.5, 1), (0.5, 0), []),                                     # device -> output init
    ("S:D", "R:3:CPU", (0.5, 0), (0.5, 1), []),                                 # build -> wavelet bank [T]
    ("R:3:CPU", "S:E", (1, 0.5), (0.5, 0), [(4, GUT / 2 + UNIT / 2, 120)]),        # bank [T] -> right, then down into E
    ("S:G", "R:6:CPU", (0.5, 0), (0.5, 1), []),                                 # allocate -> frame buffer
    ("S:I", "R:8:GPU", (0.5, 1), (0.5, 0), []),                                 # shader -> colour map (peels through [I])
    ("S:J", "R:9:GPU", (0.5, 1), (0.5, 0), []),                                 # texture alloc -> GPU texture (through [J])
]

RUNTIME_STAGES = [                       # genuinely sequential across CPU and GPU: the flow snakes through M, P, W
    (0, "CPU", "Fetch\nAudio", AUDIO, "K"),
    (1, "CPU", "FFT", DSP, "L"),
    (2, "XFER", None, DSP, "M"),
    (3, "GPU", "Freq\nDomain\nMultiply", DSP, "N"),
    (4, "GPU", "IFFT", DSP, "O"),
    (5, "GPU", "Compute\nMag", DSP, "Q"),
    (6, "GPU", "Discard\nEdges", DSP, "R"),
    (7, "GPU", "Extract\nNew Hop", DSP, "S"),
    (8, "GPU", "Down-\nsample", DSP, "T"),
    (9, "XFER", None, DSP, "P"),
    (10, "CPU", "Store\nFrame", REND, "U"),
    (11, "CPU", "Update\nTexture", REND, "V"),
    (12, "GPU", "Shader\nDraw", REND, "W"),
    (13, "CPU", "Sync\nDisplay", REND, "X"),
]
RUNTIME_RESOURCES = [
    (3, "GPU", "deck_freq", "Wavelet\nBank [F]", DSP),
    (4, "GPU", "deck_coefs", "CWT\ncoefficients", DSP),
    (5, "GPU", "morlet3_env", "magnitude", DSP),
    (6, "GPU", "mask_edges25", "edge mask", DSP),
    (7, "GPU", "scheme_hop", "new hop", DSP),
    (8, "GPU", "array4", "256:1", DSP),
    (10, "CPU", "deck_heatmap_gray", "Frame\nBuffer", INK),
    (11, "GPU", "heatmap_square_gray_col", "GPU Texture", INK),
    (12, "GPU", "cmap_vertical", "Color Map", INK, STRIP),
    (13, "GPU", "heatmap_square", "shader output", INK),
    (13, "CPU", "double_buffer", "swap\nbuffers", REND),
]
RUNTIME_HW = [(0, "CPU", "Audio\nFile\n(Disk)"), (1, "CPU", "Audio\nDevice"), (14, "GPU", "Display")]
RUNTIME_DATA = [
    ("H:0", "S:K", (0.5, 1), (0.5, 0), []),                                     # disk -> fetch
    ("H:0", "H:1", (1, 0.5), (0, 0.5), []),                                     # the same chunks play out on the device
    ("S:U", "R:10:CPU", (0.5, 0), (0.5, 1), []),                                # store -> frame buffer (host ring)
    ("R:10:CPU", "S:V", (1, 0.5), (0.5, 0), [(11, GUT / 2 + UNIT / 2, 120)]),   # frame buffer -> right, then down into update
    ("S:V", "R:11:GPU", (0.5, 1), (0.5, 0), []),                                # update -> GPU texture (through [V])
    ("R:11:GPU", "R:12:GPU", (1, 0.5), (0, 0.5), []),                           # raw texture -> color map
    ("R:12:GPU", "R:13:GPU", (1, 0.5), (0, 0.5), []),                           # color map -> colored shader output
    ("R:13:GPU", "H:14", (1, 0.5), (0, 0.5), []),                               # shader output -> display
]


def main():
    src_path, out_path = sys.argv[1], sys.argv[2]
    pages = build("Startup", STARTUP_STAGES, STARTUP_RESOURCES, STARTUP_HW, data=STARTUP_DATA) + "\n" + \
            build("Runtime", RUNTIME_STAGES, RUNTIME_RESOURCES, RUNTIME_HW, data=RUNTIME_DATA, loop_back=True)
    src = open(src_path).read()
    src = re.sub(r'\n  <diagram name="(Startup|Runtime)".*?</diagram>', "", src, flags=re.S)
    open(out_path, "w").write(src.replace("</mxfile>", pages + "\n</mxfile>"))
    print("wrote Startup + Runtime pages")


if __name__ == "__main__":
    main()
