"""Compose the Startup and Runtime swimlane diagrams from drawio primitives, both on one "Pipeline" page.

Usage:
  python tools/drawio_pipeline.py <in.drawio> <out.drawio> [split.drawio]

Rebuilds the "Pipeline" page plus "Pipeline Vertical" (both sections turned 90 degrees and stacked, Start Up above
Runtime, with an analyzer channel per letter from the timing CSVs, each section on its own time scale); every other
page is kept. A third argument writes a file with the Startup, Runtime and Pipeline Vertical pages on their own, for
the README crops.

One band for both diagrams: Start Up, a divider, then Runtime, on two lanes mirrored about the rail:
  (the runtime loop rail runs through the CPU tile row, back into the first stage from above)
  CPU lane     caption row | tile row (resources, hardware) | stage row | rail gap
  ==== rail ==== one letter box per column, bold when the column is a CPU <-> GPU transfer
  GPU lane     rail gap | stage row | tile row | caption row

  stage        rounded box coloured by domain (audio / DSP / render), joined by the solid flow line
  transfer     the flow line drops through the bold letter box; the data it carries rides the line as a
               payload tile in the sending lane's stage row (in one side, out the top/bottom)
  branch       a transfer that peels off the flow line straight into the far lane's tile row
  resource     a tile in the tile row of the lane it lives in, caption outward; a dashed arrow is data
               moving between a resource and the flow, drawn in the direction the data moves
  hardware     square ink box, dashed arrows like a resource

Stages are 2 x 2 units, gutters 1 unit. Every arrow leaves and enters a box at the centre of a side.
"""
import hashlib, re, sys
from xml.sax.saxutils import escape

sys.path.insert(0, __file__.rsplit("/", 1)[0])
sys.path.insert(0, __file__.rsplit("/", 2)[0])
import drawio_primitives as prim
from research.dsplot import style

AUDIO, DSP, REND = style.AUDIO_COLOR, style.DSP_COLOR, style.RENDER_COLOR
FONT_FAMILY = style.DRAWIO_FONT_FAMILY
INK, GRAY, LANE = "default", "#888888", "#999999"   # "default" follows the draw.io theme: black/white on light, inverted on dark

U = 40                          # design unit (draw.io grid stays 10 px)
UNIT, GUT = 2 * U, U            # stage / asset frame = 2 x 2 units, gutter = 1 unit
COL = UNIT + GUT
X0 = 180                        # first column's left gutter centre; boxes start at X0 + GUT/2 = 200
LABEL_X, LABEL_W = 40, 120      # lane titles
PAD, CAP_H, TILE_GAP, RAIL_GAP = 20, 60, 40, 50
LANE_H = PAD + CAP_H + UNIT + TILE_GAP + UNIT + RAIL_GAP     # 330: caption | tile | stage | rail gap, mirrored for the GPU
LETTER_W, LETTER_H = U, U       # letter box on the rail, one per column (stages and transfers alike)
FRAME_PAD = 10                  # inset of the content inside its UNIT cell (decks and rasters fill the cell)
STRIP = (20, 80)                # the colour map: a half-unit strip, one UNIT tall, centred in its cell
FONT = 20                       # stage labels, captions, HW labels (style.README_LABEL_EM_FRACTION, low end)
LETTER_FONT = 24                # rail letters, timing labels, tick labels
W_LINE, W_HEAVY = style.README_STROKE_DRAWIO, 2 * style.README_STROKE_DRAWIO   # every drawn line | frames, rules, bold transfers
ARC = round(100 * style.README_BOX_RADIUS)   # draw.io arcSize, % of the short side
PAYLOAD_ROW = "stage"           # "stage": a transfer's payload rides the flow line | "tile": it sits in the outer row, dashed into the drop
GPU_LANE_H = None               # None: mirror the CPU lane | a height, when the GPU outer row is unused (all tiles inline)
LOOP_ROW = "tile"               # "tile": the runtime loop rail runs through the CPU tile row | "top": along the top of the CPU lane


def snap(v, g=10):
    return int(round(v / g) * g)


def col_x(c):
    return X0 + c * COL


class Page:
    def __init__(self, name, n_cols):
        self.name, self.cells, self.late, self.n = name, [], [], 0
        self.fitted = {}                                        # tile id -> (x, y, w, h) of the drawn content
        self.width = X0 + n_cols * COL + 40
        self.n_cols = n_cols

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

    def edge(self, src, dst, color=INK, dashed=False, width=W_LINE, exit_=None, entry=None, points=(), arrow=True, src_point=None, dst_point=None, straight=False):
        """src / dst are cell ids; either may be None with a free (x, y) point instead."""
        i = self.cid()
        style = (("edgeStyle=none;rounded=0;" if straight else "edgeStyle=orthogonalEdgeStyle;rounded=1;") + f"html=1;strokeColor={color};strokeWidth={width};curved=0;"
                 + ("endArrow=block;endFill=1;endSize=6;" if arrow else "endArrow=none;")
                 + ("dashed=1;" if dashed else ""))
        if exit_:
            style += f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
        if entry:
            style += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        pts = "".join(f'<mxPoint x="{snap(px)}" y="{snap(py)}" />' for px, py in points)
        arr = f'<Array as="points">{pts}</Array>' if pts else ""
        for pt, name in ((src_point, "sourcePoint"), (dst_point, "targetPoint")):
            if pt:
                arr += f'<mxPoint x="{snap(pt[0])}" y="{snap(pt[1])}" as="{name}" />'
        ends = (f' source="{src}"' if src else "") + (f' target="{dst}"' if dst else "")
        self.cells.append(f'''
        <mxCell id="{i}" value="" style="{style}" edge="1" parent="1"{ends}>
          <mxGeometry relative="1" as="geometry">{arr}</mxGeometry>
        </mxCell>''')
        return i

    def text(self, value, x, y, w, h, size=20, color=INK, bold=True, bg=False, late=False, align="center"):
        style = (f"text;html=1;strokeColor=none;fillColor={'default' if bg else 'none'};align={align};verticalAlign=middle;whiteSpace=wrap;"
                 f"fontColor={color};fontFamily={FONT_FAMILY};fontSize={size};" + ("fontStyle=1;" if bold else "") + "labelBackgroundColor=none;")
        return self.vertex(style, x, y, w, h, value, late=late)

    def box(self, label, x, y, color, rounded=True, w=UNIT, h=UNIT, size=FONT, late=False, stroke=W_LINE, font=INK, valign="middle"):
        style = (f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;fillColor=default;strokeColor={color};fontColor={font};"
                 f"fontFamily={FONT_FAMILY};fontSize={size};fontStyle=1;align=center;verticalAlign={valign};arcSize={ARC};strokeWidth={stroke};labelBackgroundColor=none;")
        return self.vertex(style, x, y, w, h, label, late=late)

    def tile(self, key, x, y, color, caption, caption_side, size=None, width=W_LINE, backing=None):
        """A primitive fitted into a UNIT cell at (x, y), the same footprint as a stage box. Rasters (heatmaps) get a
        black frame; decks and the colour-map strip stand on their own. Caption cell above (CPU row) or below (GPU
        row). Returns (id arrows attach to, cell geometry)."""
        it = next(i for i in prim.items if i["key"] == key)
        inner = UNIT if key.startswith(("v_", "deck_", "heatmap", "cmap", "sig_", "frame_", "mask_discard", "mag_", "hop_", "downsample")) else UNIT - 2 * FRAME_PAD
        if size:
            w, h = size
        else:                                                   # fitted side snaps to 20 so the centred content stays on grid
            w, h = inner, max(20, snap(inner * it["h"] / it["w"], 20))
            if h > inner:
                h, w = inner, max(20, snap(inner * it["w"] / it["h"], 20))
        if backing:
            fill = f"fillColor={backing};"
        elif it.get("fillcolor"):
            fill = f"fillColor={it['fillcolor']};"
        elif it["fill"] or it["filled"]:
            fill = f"fillColor={color};fillOpacity=25;"
        else:
            fill = "fillColor=none;"
        gx, gy = snap(x + (UNIT - w) / 2), snap(y + (UNIT - h) / 2)
        anchor = self.box("", x, y, INK, rounded=False, w=UNIT, h=UNIT, stroke=W_LINE) if key.startswith("heatmap") else None
        if backing and not anchor:                                          # opaque backing so the tile masks whatever it rides on
            anchor = self.box("", gx, gy, "none", rounded=False, w=w, h=h, stroke=0)
        style = f"shape=stencil({prim.stencils[key]});strokeColor={color};strokeWidth={width};{fill}"
        vid = self.vertex(style, gx, gy, w, h)
        self.fitted[anchor or vid] = (gx, gy, w, h)
        if caption:
            cy = y - UNIT if caption_side == "above" else y + UNIT       # a UNIT caption cell stacked on the tile
            self.text(caption, x, cy, UNIT, UNIT, size=FONT, late=True)
        if key == "mag_345_plain":                                       # stencil text ignores the theme, so label with cells
            for label, lx, ly in (("a\u00b2", 30, 60), ("b\u00b2", 60, 30), ("c\u00b2", 20, 30)):
                self.text(label, gx + lx, gy + ly, 20, 20, size=11, late=True)
        return anchor or vid, (x, y, UNIT, UNIT)

    def attach_label(self, cid, text, side, gap=10):
        """Name a drawn cell with its own label placed outside it (left or right), so it moves with the cell."""
        align = "right" if side == "left" else "left"
        style = (f"labelPosition={side};verticalLabelPosition=middle;align={align};verticalAlign=middle;whiteSpace=wrap;html=1;"
                 f"fontFamily={FONT_FAMILY};fontSize={FONT};fontStyle=1;fontColor={INK};spacing{align.capitalize()}={gap};labelBackgroundColor=default;")
        for cells in (self.cells, self.late):
            for i, c in enumerate(cells):
                if f'id="{cid}"' in c:
                    cells[i] = (c.replace('value=""', f'value="{escape(text.replace(chr(10), "<br>"))}"', 1)
                                 .replace('" vertex="1"', style + '" vertex="1"', 1))
                    return

    def glyph(self, key, x, y, w, h, color=INK, width=W_LINE):
        """A primitive drawn at an explicit box, no frame logic (hardware contents, badges on arrows)."""
        self.box("", x, y, "none", rounded=False, w=w, h=h, stroke=0)
        return self.vertex(f"shape=stencil({prim.stencils[key]});strokeColor={color};strokeWidth={width};fillColor=none;", x, y, w, h)

    def lane(self, title, y, h, width=None):
        w = (width or self.width) - LABEL_X
        self.vertex(f"rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor={LANE};strokeWidth=1;", LABEL_X, y, w, h)
        self.text(title, LABEL_X, y, LABEL_W, h, size=22)

    def rail(self, y, width=None):
        """Double line at y-20 and y (the 'line' shape draws through the centre of its box); letter boxes centre on y-10."""
        w = (width or self.width) - LABEL_X
        for dy in (-30, -10):
            self.vertex(f"line;strokeWidth={W_LINE};html=1;strokeColor={INK};fillColor=none;", LABEL_X, y + dy, w, 20)

    def rail_box(self, letter, cx, y_div, bold=False):
        """Letter box centred on the double line at x = cx; bold (double weight) for a CPU <-> GPU transfer."""
        return self.box(letter, cx - LETTER_W / 2, y_div - 30, INK, rounded=False, w=LETTER_W, h=LETTER_H, size=LETTER_FONT, late=True, stroke=W_HEAVY if bold else W_LINE)

    def stable_ids(self, body):
        """Rename every cell id to a hash of the cell's own content (style, value, geometry, endpoints), so a
        regenerated file only adds and removes cells and never re-uses an id for a different cell: draw.io desktop
        merges a changed file into an open window by id."""
        cells = re.findall(r'<mxCell id="([^"]+)"(.*?)</mxCell>', body, flags=re.S)
        names, seen = {}, {}
        for cid, content in cells:
            key = re.sub(r'\b(source|target)="([^"]+)"', lambda m: f'{m.group(1)}="{names.get(m.group(2), m.group(2))}"', content)
            h = hashlib.sha1(key.encode()).hexdigest()[:8]
            seen[h] = seen.get(h, 0) + 1
            names[cid] = f"{self.name[:2].lower()}-{h}" + (f"-{seen[h]}" if seen[h] > 1 else "")
        return re.sub(r'\b(id|source|target)="([^"]+)"', lambda m: f'{m.group(1)}="{names.get(m.group(2), m.group(2))}"', body)

    def xml(self, height):
        body = self.stable_ids("".join(self.cells + self.late))
        return f'''  <diagram name="{self.name}" id="pipe-{self.name.lower()}">
    <mxGraphModel dx="1400" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{int(self.width) + 40}" pageHeight="{int(height)}" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />{body}
      </root>
    </mxGraphModel>
  </diagram>'''


def centre(col):
    return col_x(col) + GUT / 2 + UNIT / 2


def draw_lanes(pg, y0):
    """One CPU lane, the rail, one GPU lane, spanning the page. Returns the y of the rail (lane boundary)."""
    y_div = y0 + LANE_H
    pg.lane("CPU", y0, LANE_H)
    pg.lane("GPU", y_div, GPU_LANE_H or LANE_H)
    pg.rail(y_div)
    return y_div


def draw_section(pg, y_div, col0, stages, resources=(), hw=(), links=(), loop=False):
    """Draw one section (Start Up or Runtime) into lanes whose rail is at y_div, columns offset by col0.
       stages:    (col, lane, label, color, letter); lane CPU | GPU | XFER | BRANCH
                  XFER label: None (plain drop through the letter box) or (key, caption) for a payload tile
       resources: (col, lane, key, caption[, (w, h)[, row]])  tile in that lane's tile row (row "tile", caption
                  outward) or inline on the stage row (row "stage", caption outward). col may be "<L": the tile
                  sits one gutter left of transfer L's drop line.
       hw:        (col, lane, label[, row])              square ink box; row "tile" (default) or "stage"
       links:     (src, dst) dashed data arrows; refs 'S:<letter>' stage, 'P:<letter>' payload tile,
                  'R:<col>:<lane>' resource, 'H:<col>' hardware. Routed straight when aligned, else one bend.
       loop:      the flow returns from the last transfer along a rail in the CPU lane's tile row, into the
                  first stage from above."""
    y_cpu, y_gpu = y_div - LANE_H, y_div
    cap_y = {"CPU": y_cpu + PAD, "GPU": y_gpu + RAIL_GAP + UNIT + TILE_GAP + UNIT}
    tile_y = {"CPU": y_cpu + PAD + CAP_H, "GPU": y_gpu + RAIL_GAP + UNIT + TILE_GAP}
    box_y = {"CPU": y_div - RAIL_GAP - UNIT, "GPU": y_gpu + RAIL_GAP}
    mid = {k: v + UNIT / 2 for k, v in box_y.items()}
    cx_of = lambda col: centre(col0 + col)
    left_of = lambda col: col_x(col0 + col) + GUT / 2
    letter_col = {st[4]: st[0] for st in stages}

    ids, geom = {}, {}                                          # ref -> cell id, ref -> (x, y, w, h)

    def place(ref, cid, x, y, w=UNIT, h=UNIT):
        ids[ref], geom[ref] = cid, (x, y, w, h)

    for i, (col, lane, label, color, letter) in enumerate(stages):
        cx = cx_of(col)
        pg.rail_box(letter, cx, y_div, bold=lane in ("XFER", "BRANCH"))
        if lane in ("CPU", "GPU"):
            place(f"S:{letter}", pg.box(label, cx - UNIT / 2, box_y[lane], color), cx - UNIT / 2, box_y[lane])
        elif lane == "XFER" and label:                          # payload rides the flow in the sending lane's stage row
            key, caption = label
            side = stages[i - 1][1]
            outboard = PAYLOAD_ROW == "tile"
            x, y = cx - UNIT / 2, tile_y[side] if outboard else box_y[side]
            tid, _ = pg.tile(key, x, y, INK, None, None)
            place(f"P:{letter}", tid, x, y)
            cy = cap_y[side] if outboard else (y - CAP_H if side == "CPU" else y + UNIT)
            pg.text(caption, x - GUT / 2, cy, UNIT + GUT, CAP_H, size=FONT, bg=True, late=True)
            if outboard:                                        # data joins the flow at the drop corner
                pg.edge(tid, None, dashed=True, exit_=(0.5, 1 if side == "CPU" else 0), dst_point=(cx, mid[side]))

    for entry in resources:
        col, lane, key, caption = entry[:4]
        size = entry[4] if len(entry) > 4 else None
        row = entry[5] if len(entry) > 5 else "tile"
        if isinstance(col, str):                                # "<L": content ends one gutter left of L's drop line
            w = size[0] if size else UNIT
            x = snap(cx_of(letter_col[col[1:]]) - GUT - w - (UNIT - w) / 2)
        else:
            x = left_of(col)
        y = box_y[lane] if row == "stage" else tile_y[lane]
        tid, _ = pg.tile(key, x, y, INK, None, None, size=size)
        place(f"R:{col}:{lane}", tid, x, y)
        cy = cap_y[lane] if row == "tile" else (y - CAP_H if lane == "CPU" else y + UNIT)
        pg.text(caption, x - GUT / 2, cy, UNIT + GUT, CAP_H, size=FONT, bg=True, late=True)

    for h in hw:
        col, lane, label = h[:3]
        row = h[3] if len(h) > 3 else "tile"
        x, y = left_of(col), (box_y if row == "stage" else tile_y)[lane]
        place(f"H:{col}", pg.box(label, x, y, INK, rounded=False), x, y)

    # the flow line
    first = next(s for s in stages if s[1] in ("CPU", "GPU"))
    y_rail = tile_y["CPU"] + UNIT / 2 if LOOP_ROW == "tile" else y_cpu + PAD / 2     # loop rail, down into the first stage
    x_first = cx_of(first[0])
    i = 0
    while i + 1 < len(stages):
        col, lane, label, color, letter = stages[i]
        nxt = stages[i + 1]
        if nxt[1] in ("CPU", "GPU"):
            pg.edge(ids[f"S:{letter}"], ids[f"S:{nxt[4]}"], exit_=(1, 0.5), entry=(0, 0.5))
            i += 1
            continue
        xcx = cx_of(nxt[0])
        after = stages[i + 2] if i + 2 < len(stages) else None
        if nxt[1] == "BRANCH":                                  # peel off the line, straight into the far lane's tile
            far = "GPU" if lane == "CPU" else "CPU"
            pg.edge(ids[f"S:{letter}"], ids[f"R:{nxt[0]}:{far}"], exit_=(1, 0.5),
                    entry=(0.5, 0) if far == "GPU" else (0.5, 1), points=[(xcx, mid[lane])])
            if after:
                pg.edge(ids[f"S:{letter}"], ids[f"S:{after[4]}"], exit_=(1, 0.5), entry=(0, 0.5))
            i += 2
            continue
        to_lane = after[1] if after else "CPU"
        down = to_lane == "GPU"
        if nxt[2] and PAYLOAD_ROW == "stage":                   # through the payload tile, then through the letter box
            tid = ids[f"P:{nxt[4]}"]
            pg.edge(ids[f"S:{letter}"], tid, exit_=(1, 0.5), entry=(0, 0.5))
            src, exit_, start = tid, (0.5, 1 if down else 0), []
        else:                                                   # plain drop through the letter box
            src, exit_, start = ids[f"S:{letter}"], (1, 0.5), [(xcx, mid[lane])]
        if after:
            pg.edge(src, ids[f"S:{after[4]}"], exit_=exit_, entry=(0, 0.5), points=start + [(xcx, mid[to_lane])])
        else:                                                   # last transfer: up onto the loop rail and back to the start
            pg.edge(src, ids[f"S:{first[4]}"], exit_=exit_, entry=(0.5, 0),
                    points=start + [(xcx, y_rail), (x_first, y_rail)])
        i += 2

    for src, dst in links:                                      # dashed: data moving between a resource and the flow
        (sx, sy, sw, sh), (dx, dy, dw, dh) = geom[src], geom[dst]
        scx, dcx = sx + sw / 2, dx + dw / 2
        if scx == dcx:                                          # same column: straight up or down
            up = dy < sy
            pg.edge(ids[src], ids[dst], dashed=True, exit_=(0.5, 0 if up else 1), entry=(0.5, 1 if up else 0))
        elif sy == dy:                                          # same row: straight across
            pg.edge(ids[src], ids[dst], dashed=True, exit_=(1, 0.5), entry=(0, 0.5))
        else:                                                   # out the side, one bend, in the top or bottom
            up = dy < sy
            pg.edge(ids[src], ids[dst], dashed=True, exit_=(1, 0.5), entry=(0.5, 1 if up else 0), points=[(dcx, sy + sh / 2)])


def draw_divider(pg, y_div, col):
    """A light vertical line through the gutter column between two sections."""
    x = centre(col)
    pg.edge(None, None, color=LANE, width=1, arrow=False, dashed=True,
            src_point=(x, y_div - LANE_H), dst_point=(x, y_div + LANE_H))


STARTUP = dict(
    stages=[                             # all host code: allocations peel off the flow line, nothing runs on the GPU in sequence
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
    ],
    resources=[
        (3, "CPU", "deck_wavelets", "Wavelet\nBank [T]"),
        (5, "GPU", "deck_freq", "Wavelet\nBank [F]", None, "stage"),     # the F branch lands on it, inline
        (6, "CPU", "v_deck_init", "Frame\nBuffer"),                      # deck, init
        (8, "GPU", "cmap_vertical", "Color Map", STRIP, "stage"),
        (9, "GPU", "v_texture_init", "GPU\nTexture", None, "stage"),     # texture, init
    ],
    hw=[(0, "CPU", "Audio\nFile\n(Disk)"), (1, "CPU", "Audio\nDevice")],
    links=[
        ("H:0", "S:A"),                  # disk -> open
        ("H:1", "S:B"),                  # device -> output init
        ("S:D", "R:3:CPU"),              # build -> wavelet bank [T]
        ("R:3:CPU", "S:E"),              # bank [T] -> right, then down into E
        ("S:G", "R:6:CPU"),              # allocate -> frame buffer
        ("S:I", "R:8:GPU"),              # shader -> colour map
        ("S:J", "R:9:GPU"),              # texture alloc -> GPU texture
    ],
)

RUNTIME = dict(
    stages=[                             # every lane change is a bold letter box carrying its payload: M, P, V, X
        (1, "CPU", "Fetch\nAudio", AUDIO, "K"),
        (2, "CPU", "FFT", DSP, "L"),
        (3, "XFER", ("sig_spectrum", "Input\nSignal [F]"), DSP, "M"),
        (4, "GPU", "Freq\nDomain\nMultiply", DSP, "N"),
        (5, "GPU", "IFFT", DSP, "O"),
        (6, "GPU", "Post\nProcess", DSP, "Q"),      # magnitude, discard edges, trim overlap, downsample
        (7, "XFER", ("v_frame_raw", "CWT\nFrame"), DSP, "P"),                 # single frame, raw
        (8, "CPU", "Store\nFrame", REND, "U"),
        (9, "XFER", ("v_deck_raw", "Frame\nBuffer"), REND, "V"),              # deck, raw (the CPU side keeps the deck)
        (10, "GPU", "Shader\nDraw", REND, "W"),
        (11, "XFER", ("v_texture_color", "Spectrogram\nTexture"), REND, "X"),  # texture, color mapped
    ],
    resources=[                          # inline on the stage row, one gutter left of the drop that feeds their consumer
        (0, "CPU", "sig_noise", "Input\nSignal [T]", None, "stage"),      # the next chunk, consumed by K
        ("<M", "GPU", "deck_freq", "Wavelet\nBank [F]", None, "stage"),
        ("<V", "GPU", "cmap_vertical", "Color Map", STRIP, "stage"),
    ],
    hw=[(12, "GPU", "Display", "stage")],
    links=[("R:0:CPU", "S:K"), ("R:<M:GPU", "S:N"), ("R:<V:GPU", "S:W"), ("P:X", "H:12")],
    loop=True,
)


def shifted(spec, d, **over):
    """A copy of a section spec with every row index moved by d (string rows like "<M" are relative and stay)."""
    def sh(r):
        return r if isinstance(r, str) else r + d
    out = dict(spec)
    for k in ("stages", "resources", "hw"):
        out[k] = [(sh(e[0]),) + tuple(e[1:]) for e in spec[k]]
    out["links"] = [tuple(re.sub(r"^([RH]):(\d+)", lambda m: f"{m.group(1)}:{int(m.group(2)) + d}", ref) for ref in link)
                    for link in spec["links"]]
    out.update(over)
    return out


# The vertical page's runtime: Input Signal [T] sits beside Fetch Audio on K's row like the start-up hardware boxes,
# so the section has no row before K and the loop enters Fetch Audio from above.
# The vertical page's runtime: GPU resources in the GPU tile column level with the stage they feed (a straight dashed
# arrow across, labels outward like the CPU's), Input Signal [T] on the band row (level with the whole-cycle pulse: the
# chunk the cycle consumes), the loop up the CPU tile column.
# Input Signal [T] (the next chunk) sits beside Fetch Audio in the CPU tile column like the start-up hardware boxes,
# and the loop returns into it: the cycle comes back for the next chunk, which feeds Fetch Audio.
RUNTIME_V = shifted(dict(RUNTIME,
                         resources=[("=K", "CPU", "sig_noise", "Input\nSignal [T]"),
                                    ("=N", "GPU", "deck_freq", "Wavelet\nBank [F]"),
                                    ("=W", "GPU", "cmap_vertical", "Color Map", STRIP)],
                         links=[("R:=K:CPU", "S:K"), ("R:=N:GPU", "S:N"), ("R:=W:GPU", "S:W"), ("P:X", "H:12")]),
                    -1, loop_entry="chunk")


# ---------------------------------------------------------------------------------------------------------------------
# Vertical pages: the same sections turned 90 degrees (columns become rows, "above" becomes "left of"), with an
# analyzer channel per letter beside them: a row is a time slot, its pulse is the stage's measured time.
#   CPU lane (left)   label room | tile col | stage col | rail gap
#   ==== rail ==== vertical double line, one letter box per row, bold on transfers
#   GPU lane (right)  rail gap | stage col | label room        (no tile column: every GPU resource is inline)
# Resource names are the glyph's own draw.io label, set outboard of it (labelPosition), so they move with the symbol.
# draw_section_v is draw_section with x and y swapped; every exit / entry fraction and waypoint is transposed.

TIMING_RUN = "20260830_183812_GpuCWT_c16384_f116"
TIMING_DIR = __file__.rsplit("/", 2)[0] + "/assets/timing"
LETTER_STAGES = {                        # letter -> the timing report's stage keys it covers
    "A": ("init:audio_reader",), "B": ("init:audio_player",), "C": ("init:cuda",), "D": ("init:dsp_kernels",),
    "E": ("init:dsp_fft",), "F": ("init:dsp_upload",), "G": ("init:render_buffer",), "H": ("init:render_glcontext",),
    "I": ("init:render_shader",), "J": (),          # shader + texture are one timed block: I carries it, J reads "I + J"
    "K": ("audio_read",), "L": ("fft_cpu",), "M": ("upload",), "N": ("multiply",), "O": ("ifft",),
    "Q": ("magnitude", "edge_trim", "hop_center", "downsample"), "P": ("download",), "U": ("buf_push",),
    "V": ("tex_upload",), "W": ("gl_clear", "gl_draw"), "X": ("gl_swap",)}
STARTUP_TOTAL_KEYS = ("init:audio", "init:cuda", "init:dsp", "init:renderer", "init:prescan", "init:prime", "init:other")
PANEL = dict(Startup=dict(axis_max=800, step=100, px_per=1.2, fmt="{:.0f}"),     # 800 ms and 6 ms both span SPAN;
             Runtime=dict(axis_max=8, step=1, px_per=120, fmt="{:.2f}"))         # ticks land on 120 / 160 px
Y_PAGE = 40                              # page top
SECTION_H, BAND_H, TITLE_FONT = COL, COL, 40   # section name band spanning lanes + analyzer | CPU/GPU band with the whole-section pulse
Y_LABEL = Y_PAGE + SECTION_H             # lane top (mirrors LABEL_X): the first section's frame hangs off its name band
Y0 = Y_LABEL + BAND_H                    # rows (mirrors X0): every band is COL tall, its cell centred (a half gutter each side)
V_RAIL_GAP, V_ROOM = 80, 160                                  # rail gap (gutter + half rail cell) | label room: gutter + label cell + gutter
V_LANE_CPU = V_ROOM + UNIT + TILE_GAP + UNIT + V_RAIL_GAP     # 380: label room | tile col | stage col | rail gap
V_LANE_GPU = V_LANE_CPU                                       # mirrored: rail gap | stage col | tile col | label room
SUBLANES = True                                               # ghost line between the stage and tile columns of each lane
V_MARGIN = 80                            # left of the CPU lane (room for a loop outside it)
X_DIV = V_MARGIN + V_LANE_CPU            # the rail
SPAN = 960                               # analyzer span (0 to axis_max)
PANEL_PAD = GUT                          # padding either side of the span inside the frame, room for the end tick labels
GAP_PANEL = PANEL_PAD                    # the analyzer shares the lanes' frame: its span starts one pad past the GPU lane
H_GRID = "all"                         # ghost lines on the row boundaries: None | "panel" | "all" (across the lanes too)
RAIL_INK, FLOW_W = INK, W_LINE
BORDER_INK, BORDER_W = INK, W_HEAVY                                   # lane / panel frames and the section divider
T_FONT = LETTER_FONT                                            # timing labels, same size as the rail letters
LOOP_INK, LOOP_W = INK, W_LINE                # the loop's return when it runs outside the lane


def load_timings():
    """Per-letter ms and section totals for TIMING_RUN: start up from the results table (mean), runtime from the
    per-frame iterations (median). Returns ({letter: ms | None}, {"Startup": total, "Runtime": total})."""
    import csv, statistics
    results = {r["stage"]: float(r["mean_ms"]) for r in csv.DictReader(open(f"{TIMING_DIR}/timing_results.csv"))
               if r["run_id"] == TIMING_RUN}
    frames = {}
    for r in csv.DictReader(open(f"{TIMING_DIR}/timing_iterations.csv")):
        if r["run_id"] == TIMING_RUN:
            frames.setdefault(r["stage"], []).append(float(r["ms"]))
    medians = {k: statistics.median(v) for k, v in frames.items()}
    ms = {}
    for letter, keys in LETTER_STAGES.items():
        table = results if letter <= "J" else medians
        ms[letter] = sum(table[k] for k in keys) if keys else None
    return ms, {"Startup": sum(results[k] for k in STARTUP_TOTAL_KEYS), "Runtime": medians["TOTAL"]}


def row_c(r):                            # centre y of row r (mirrors centre(col))
    return Y0 + r * COL + GUT / 2 + UNIT / 2


def row_top(r):
    return Y0 + r * COL + GUT / 2


def transpose(p):
    return (p[1], p[0])


def draw_lanes_v(pg, y_end, title_ys, spans):
    """One frame round the lanes and the analyzer from Y_LABEL to y_end + GUT / 2, a heavy line where the GPU lane meets
    the analyzer, a large centred title per lane in each section's band; the rail (centred on X_DIV, lines at +-10,
    boxes +-20) and the sub-lane dividers run only through each section's rows (the y spans)."""
    y1 = y_end + GUT / 2
    x_left, x_right = X_DIV - V_LANE_CPU, X_DIV + V_LANE_GPU + GAP_PANEL + SPAN + PANEL_PAD
    pg.vertex(f"rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor={BORDER_INK};strokeWidth={BORDER_W};",
              x_left, Y_LABEL, x_right - x_left, y1 - Y_LABEL)
    pg.edge(None, None, color=BORDER_INK, width=BORDER_W, arrow=False, src_point=(X_DIV + V_LANE_GPU, Y_LABEL), dst_point=(X_DIV + V_LANE_GPU, y1))
    for x, w, name in ((X_DIV - V_LANE_CPU, V_LANE_CPU, "CPU"), (X_DIV, V_LANE_GPU, "GPU")):
        for y in title_ys:
            pg.text(name, x, y, w, BAND_H, size=TITLE_FONT)
    for ya, yb in spans:
        if SUBLANES:
            for sx in (X_DIV - V_RAIL_GAP - UNIT - TILE_GAP / 2, X_DIV + V_RAIL_GAP + UNIT + TILE_GAP / 2):
                pg.edge(None, None, color=LANE, width=1, arrow=False, dashed=True, src_point=(sx, ya), dst_point=(sx, yb))
        for dx in (-10, 10):
            pg.vertex(f"line;strokeWidth={W_LINE};html=1;strokeColor={RAIL_INK};fillColor=none;direction=south;", X_DIV + dx - 10, ya, 20, yb - ya)


def rail_box_v(pg, letter, cy, bold):
    return pg.box(letter, X_DIV - 20, cy - 20, INK, rounded=False, w=LETTER_W, h=LETTER_H, size=LETTER_FONT, late=True, stroke=W_HEAVY if bold else W_LINE)


def edge_v(pg, src, dst, exit_=None, entry=None, points=(), **kw):
    """pg.edge with the horizontal section's exit / entry fractions and waypoints transposed."""
    return pg.edge(src, dst, exit_=transpose(exit_) if exit_ else None, entry=transpose(entry) if entry else None,
                   points=[transpose(p) for p in points], **kw)


def draw_section_v(pg, stages, resources=(), hw=(), links=(), loop=False, loop_entry="left", dy=0):
    x_cpu, x_gpu = X_DIV - V_LANE_CPU, X_DIV
    tile_x = {"CPU": x_cpu + V_ROOM, "GPU": x_gpu + V_RAIL_GAP + UNIT + TILE_GAP}
    box_x = {"CPU": X_DIV - V_RAIL_GAP - UNIT, "GPU": x_gpu + V_RAIL_GAP}
    mid = {k: v + UNIT / 2 for k, v in box_x.items()}
    letter_row = {st[4]: st[0] for st in stages}
    stage_lane = {st[4]: st[1] for st in stages}
    uploads = {a[2:] for a, b in links                          # a stage whose data link crosses to the other lane
               if a.startswith("S:") and b.startswith("R:") and b.split(":")[2] != stage_lane[a[2:]]}
    ids, geom = {}, {}

    def rc(r):                                                  # section rows are relative; dy stacks sections
        return row_c(r) + dy

    def rt(r):
        return row_top(r) + dy

    def place(ref, cid, x, y, w=UNIT, h=UNIT):
        ids[ref], geom[ref] = cid, (x, y, w, h)

    def caption(text, tid, lane):                               # the glyph's own label, in the label cell one gutter outboard
        pg.attach_label(tid, text, "left" if lane == "CPU" else "right", gap=GUT)

    for i, (row, lane, label, color, letter) in enumerate(stages):
        cy = rc(row)
        rail_box_v(pg, letter, cy, bold=lane in ("XFER", "BRANCH") or letter in uploads)
        if lane in ("CPU", "GPU"):
            place(f"S:{letter}", pg.box(label, box_x[lane], cy - UNIT / 2, color), box_x[lane], cy - UNIT / 2)
        elif lane == "XFER" and label:
            key, cap = label
            side = stages[i - 1][1]
            x, y = box_x[side], cy - UNIT / 2
            tid, _ = pg.tile(key, x, y, INK, None, None)
            place(f"P:{letter}", tid, x, y)
            caption(cap, tid, side)

    for entry in resources:
        row, lane, key, cap = entry[:4]
        size = entry[4] if len(entry) > 4 else None
        rk = entry[5] if len(entry) > 5 else "tile"
        h = size[1] if size else UNIT
        r = letter_row[row[1:]] - (row[0] == "<") if isinstance(row, str) else row   # "<L": the row above L; "=L": L's own row
        y = rt(r) + (UNIT - h) / 2
        x = box_x[lane] if rk == "stage" else tile_x[lane]
        tid, _ = pg.tile(key, x, y, INK, None, None, size=size)
        place(f"R:{row}:{lane}", tid, x, y)
        caption(cap, tid, lane)

    for h in hw:
        row, lane, label = h[:3]
        rk = h[3] if len(h) > 3 else "tile"
        x, y = (box_x if rk == "stage" else tile_x)[lane], rt(row)
        place(f"H:{row}", pg.box(label, x, y, INK, rounded=False), x, y)

    # the flow line: every exit / entry below is the horizontal section's, transposed by edge_v
    first = next(s for s in stages if s[1] in ("CPU", "GPU"))
    x_rail = tile_x["CPU"] if loop_entry == "left" else x_cpu + 20               # loop rail: the CPU tile column's outer edge | the lane edge
    y_first = rc(first[0])
    i = 0
    while i + 1 < len(stages):
        row, lane, label, color, letter = stages[i]
        nxt = stages[i + 1]
        if nxt[1] in ("CPU", "GPU"):
            edge_v(pg, ids[f"S:{letter}"], ids[f"S:{nxt[4]}"], exit_=(1, 0.5), entry=(0, 0.5))
            i += 1
            continue
        ycy = rc(nxt[0])
        after = stages[i + 2] if i + 2 < len(stages) else None
        if nxt[1] == "BRANCH":
            far = "GPU" if lane == "CPU" else "CPU"
            edge_v(pg, ids[f"S:{letter}"], ids[f"R:{nxt[0]}:{far}"], exit_=(1, 0.5),
                   entry=(0.5, 0) if far == "GPU" else (0.5, 1), points=[(ycy, mid[lane])])
            if after:
                edge_v(pg, ids[f"S:{letter}"], ids[f"S:{after[4]}"], exit_=(1, 0.5), entry=(0, 0.5))
            i += 2
            continue
        to_lane = after[1] if after else "CPU"
        down = to_lane == "GPU"
        if nxt[2]:
            tid = ids[f"P:{nxt[4]}"]
            edge_v(pg, ids[f"S:{letter}"], tid, exit_=(1, 0.5), entry=(0, 0.5))
            src, exit_, start = tid, (0.5, 1 if down else 0), []
        else:
            src, exit_, start = ids[f"S:{letter}"], (1, 0.5), [(ycy, mid[lane])]
        if after:
            edge_v(pg, src, ids[f"S:{after[4]}"], exit_=exit_, entry=(0, 0.5), points=start + [(ycy, mid[to_lane])])
        elif not loop:                                          # last transfer: the flow ends at its payload
            pass
        elif loop_entry == "rail":                              # last transfer: into the rail, up inside it, out at K
            edge_v(pg, src, ids[f"S:{first[4]}"], exit_=exit_, entry=(0.5, 1),
                   points=start + [(ycy, X_DIV), (y_first, X_DIV)], width=FLOW_W)
        elif loop_entry == "outside":                           # last transfer: out past the lane border, up the margin, in from above
            y_g = y_first - UNIT / 2 - GUT / 2
            xo = x_cpu - GUT
            edge_v(pg, src, ids[f"S:{first[4]}"], exit_=exit_, entry=(0, 0.5),
                   points=start + [(ycy, xo), (y_g, xo), (y_g, mid["CPU"])], color=LOOP_INK, width=LOOP_W)
        elif loop_entry == "badge":                             # last transfer: the flow ends at a "back to K" box in the idle CPU column
            bid = pg.box("\u21ba" + first[4], box_x["CPU"] + (UNIT - LETTER_W) / 2, ycy - LETTER_H / 2, INK,
                         rounded=False, w=LETTER_W, h=LETTER_H, size=LETTER_FONT, late=True, stroke=W_LINE)
            edge_v(pg, src, bid, exit_=exit_, entry=(0.5, 1), points=start)
        elif loop_entry == "chunk":                             # last transfer: out past the tiles, up the gutter, into the next chunk's left side
            xr = tile_x["CPU"] - GUT / 2
            edge_v(pg, src, ids[f"R:={first[4]}:CPU"], exit_=exit_, entry=(0.5, 0),
                   points=start + [(ycy, xr), (y_first, xr)])
        elif loop_entry == "top":                               # last transfer: out to the loop rail, up, in from above
            y_g = y_first - UNIT / 2 - GUT / 2
            edge_v(pg, src, ids[f"S:{first[4]}"], exit_=exit_, entry=(0, 0.5),
                   points=start + [(ycy, x_rail), (y_g, x_rail), (y_g, mid["CPU"])])
        else:                                                   # last transfer: out to the loop rail and back to the start
            edge_v(pg, src, ids[f"S:{first[4]}"], exit_=exit_, entry=(0.5, 0),
                   points=start + [(ycy, x_rail), (y_first, x_rail)])
        i += 2

    for src, dst in links:                                      # dashed: data moving between a resource and the flow
        (sx, sy, sw, sh), (dx, dy, dw, dh) = geom[src], geom[dst]
        scy, dcy = sy + sh / 2, dy + dh / 2
        left = dx < sx
        if scy == dcy:                                          # same row: straight across
            pg.edge(ids[src], ids[dst], dashed=True, exit_=(0 if left else 1, 0.5), entry=(1 if left else 0, 0.5))
        elif sx == dx:                                          # same column: straight down
            pg.edge(ids[src], ids[dst], dashed=True, exit_=(0.5, 1), entry=(0.5, 0))
        else:                                                   # out the bottom, one bend, in the left or right side
            pg.edge(ids[src], ids[dst], dashed=True, exit_=(0.5, 1), entry=(1 if left else 0, 0.5), points=[(sx + sw / 2, dcy)])


def draw_panel(pg, name, stages, band_y, y_start, y_end, ms, total, dy=0):
    """Analyzer channels right of the GPU lane, on the section's own time scale: the whole section pulses on the band
    row at band_y, one pulse per letter on its row, the axis under the last row."""
    p = PANEL[name]
    x0, px = X_DIV + V_LANE_GPU + GAP_PANEL, p["px_per"]
    axis_y = y_end - GUT / 2
    t = 0
    while t <= p["axis_max"]:
        gx = x0 + t * px
        pg.edge(None, None, color=LANE, width=1, arrow=False, src_point=(gx, y_start + BAND_H), dst_point=(gx, axis_y))   # from the rule under the band
        pg.text(f"{t:g}{' ms' if t == p['axis_max'] else ''}", gx - 60, axis_y, 120, GUT, size=T_FONT, late=True)
        t += p["step"]
    pg.edge(None, None, arrow=False, src_point=(x0, axis_y), dst_point=(x0 + SPAN, axis_y))
    if H_GRID:
        gx0 = X_DIV - V_LANE_CPU if H_GRID == "all" else x0
        y = y_start + BAND_H + COL                              # row boundaries (gutter centres) under row 0
        while y < axis_y:
            pg.edge(None, None, color=LANE, width=1, arrow=False, src_point=(gx0, y), dst_point=(x0 + SPAN + PANEL_PAD, y))
            y += COL

    def pulse(y_top, xs, xe, color, fill, fill_color=None):
        xs, xe = snap(xs), snap(xe)                             # fill and trace share the snapped edges
        if xe > xs:
            pg.vertex(f"rounded=0;html=1;fillColor={fill_color or color};fillOpacity={fill};strokeColor=none;", xs, y_top, xe - xs, UNIT)
        pts = [(x0, y_top + UNIT), (xs, y_top + UNIT), (xs, y_top), (xe, y_top), (xe, y_top + UNIT), (x0 + SPAN, y_top + UNIT)]
        pg.edge(None, None, color=color, arrow=False, points=pts[1:-1], src_point=pts[0], dst_point=pts[-1], straight=True)

    def label(y_top, xs, xe, txt):
        w = max(80, len(txt) * T_FONT * 0.62)
        cx = (xs + xe) / 2 if xe - xs > w + 10 else (xs - w / 2 - 10 if xe > x0 + SPAN - w - 20 else xe + w / 2 + 10)
        pg.text(txt, cx - w / 2, y_top, w, UNIT, size=T_FONT, late=True)

    fmt = p["fmt"]
    t = t_prev = 0.0
    prev = None
    for row, lane, _label, color, letter in stages:
        y_top = row_top(row) + dy
        if ms[letter] is None:                                  # timed together with the row above
            pg.edge(None, None, color=color, arrow=False, src_point=(x0, y_top + UNIT), dst_point=(x0 + SPAN, y_top + UNIT), straight=True)
            label(y_top, x0 + t_prev * px, x0 + t_prev * px, f"{prev} + {letter}")
            continue
        xs, xe = x0 + t * px, x0 + (t + ms[letter]) * px
        pulse(y_top, xs, xe, color, 20)
        label(y_top, xs, xe, (fmt.format(ms[letter]) if ms[letter] >= 1 or "2f" in fmt else f"{ms[letter]:.1f}") + " ms")
        t_prev, t, prev = t, t + ms[letter], letter
    pulse(band_y, x0, x0 + total * px, INK, 40, fill_color=GRAY)                # a theme-colour fill would export black
    pg.text(fmt.format(total) + " ms", x0 + total * px / 2 - 60, band_y, 120, UNIT, size=T_FONT, late=True)
    return x0 + SPAN + PANEL_PAD + V_MARGIN


def n_rows(spec):
    return max(c[0] for k in ("stages", "resources", "hw") for c in spec[k] if not isinstance(c[0], str)) + 1


SECTIONS_V = (("Start Up", "Startup", STARTUP), ("Runtime", "Runtime", RUNTIME_V))   # band title, PANEL / totals key, spec


def pipeline_vertical_page(ms, totals, sections=SECTIONS_V, name="Pipeline Vertical"):
    """The sections stacked on one pair of lanes. Each hangs off its top line (the lane top, then a divider one gutter
    under the previous section's tick labels) the same way: a section name band across lanes and analyzer, a band
    cell with the lane titles and the whole-section pulse, then its rows."""
    pg = Page(name, 1)
    x_panel = X_DIV + V_LANE_GPU + GAP_PANEL
    x_left, x_right = X_DIV - V_LANE_CPU, x_panel + SPAN + PANEL_PAD
    y_top, placed = Y_LABEL, []
    for title, key, spec in sections:
        dy = y_top - Y_LABEL                                    # rows hang off this section's top as Start Up's off the page top
        y_axis = Y0 + dy + n_rows(spec) * COL                   # axis at y_axis - 20, tick labels to y_axis + 20
        draw_section_v(pg, dy=dy, **spec)
        placed.append((title, key, spec, y_top, y_axis, dy))
        y_top = y_axis + GUT + SECTION_H
    y_end = placed[-1][4]
    spans = [(y_top + BAND_H, y_axis + GUT) for _, _, _, y_top, y_axis, _ in placed[:-1]] + [(placed[-1][3] + BAND_H, y_end + GUT / 2)]
    draw_lanes_v(pg, y_end, [y_top for _, _, _, y_top, _, _ in placed], spans)
    for title, key, spec, y_top, y_axis, dy in placed:
        pg.edge(None, None, color=BORDER_INK, width=BORDER_W, arrow=False,      # rule closing the CPU/GPU band, full width
                src_point=(x_left, y_top + BAND_H), dst_point=(x_right, y_top + BAND_H))
        pg.box(title, x_left, y_top - SECTION_H, BORDER_INK, rounded=False, w=x_right - x_left, h=SECTION_H,
               size=TITLE_FONT, late=True, stroke=W_HEAVY)
        pg.width = draw_panel(pg, key, spec["stages"], y_top + GUT / 2, y_top, y_axis, ms, totals[key], dy=dy)
    return pg.xml(y_end + GUT / 2 + 40)


SECTION_GAP = 1                          # empty columns between the Start Up and Runtime sections


def main():
    src_path, out_path = sys.argv[1], sys.argv[2]
    startup_cols = max(max(c[0] for c in STARTUP[k] if not isinstance(c[0], str)) for k in ("stages", "resources", "hw")) + 1
    runtime_col0 = startup_cols + SECTION_GAP
    runtime_cols = max(max(c[0] for c in RUNTIME[k] if not isinstance(c[0], str)) for k in ("stages", "resources", "hw")) + 1
    pg = Page("Pipeline", runtime_col0 + runtime_cols)
    y_div = draw_lanes(pg, 20)
    draw_section(pg, y_div, 0, **STARTUP)
    draw_divider(pg, y_div, startup_cols)
    draw_section(pg, y_div, runtime_col0, **RUNTIME)
    ms, totals = load_timings()
    vertical = [pipeline_vertical_page(ms, totals)]
    pages = "\n".join([pg.xml(y_div + (GPU_LANE_H or LANE_H) + 20)] + vertical)
    src = open(src_path).read()
    src = re.sub(r'\n  <diagram name="(Startup|Runtime|Pipeline|Startup Vertical|Start Up Vertical|Runtime Vertical|Pipeline Vertical)".*?</diagram>', "", src, flags=re.S)
    open(out_path, "w").write(src.replace("</mxfile>", pages + "\n</mxfile>"))
    print("wrote Pipeline, Pipeline Vertical pages")
    if len(sys.argv) > 3:                # optional: a file with each section on its own page, for the README crops
        pages = "\n".join([section_page(name, cols, spec) for name, cols, spec in
                           (("Startup", startup_cols, STARTUP), ("Runtime", runtime_cols, RUNTIME))] + vertical
                          + [pipeline_vertical_page(ms, totals, (sec,), f"{sec[0]} Vertical") for sec in SECTIONS_V])
        open(sys.argv[3], "w").write(f'<mxfile host="drawio_pipeline.py">\n{pages}\n</mxfile>')
        print("wrote Startup, Runtime, Pipeline Vertical, Start Up Vertical, Runtime Vertical pages to", sys.argv[3])


def section_page(name, n_cols, spec):
    pg = Page(name, n_cols)
    y_div = draw_lanes(pg, 20)
    draw_section(pg, y_div, 0, **spec)
    return pg.xml(y_div + (GPU_LANE_H or LANE_H) + 20)


if __name__ == "__main__":
    main()
