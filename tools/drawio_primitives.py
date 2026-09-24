"""Generate draw.io primitives for SubShader diagrams.

Usage:
  python tools/drawio_primitives.py <in.drawio> <out.drawio> <library.xml>

Keeps every page of <in.drawio> except the generated "Primitives" page, which is
rebuilt. Also writes a draw.io shape library (File > Open Library).

Waves are stencil shapes built from cubic Beziers with exact tangents, so they
stay smooth at any size. Every primitive is a group: border box + stencil (+ labels).
"""
import base64, json, math, re, sys, urllib.parse, zlib
from xml.sax.saxutils import escape

TILE_W, TILE_H = 160, 80
AMP = 0.40 * TILE_H
MID = TILE_H / 2
COL_STEP, ROW_GAP = 200, 40
ORIGIN_X, ORIGIN_Y = 160, 80
HEADER_H = 40
STROKE = "strokeColor=default;strokeWidth=1.5;"       # "default" follows the draw.io theme
FILL = "fillColor=#808080;fillOpacity=30;"            # mid gray reads on light and dark canvases
FREQS = [1, 2, 3, 4, 5]
SEGS = 8
MORLET_SIGMA_CYCLES = 0.75
WIDE_SIGMA = 0.18


def compress(text):
    enc = urllib.parse.quote(text, safe="-_.!~*'()").encode()
    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    return base64.b64encode(c.compress(enc) + c.flush()).decode()


def hermite_path(f, n_segs, x0=0.0, x1=TILE_W):
    eps = 1e-4
    def df(t):
        a, b = max(0.0, t - eps), min(1.0, t + eps)
        return (f(b) - f(a)) / (b - a)
    span = x1 - x0
    ops = [f'<move x="{x0:.2f}" y="{f(0):.2f}"/>']
    for i in range(n_segs):
        t0, t1 = i / n_segs, (i + 1) / n_segs
        xa, xb = x0 + t0 * span, x0 + t1 * span
        dx = xb - xa
        ya, yb = f(t0), f(t1)
        da, db = df(t0) / span, df(t1) / span
        ops.append(f'<curve x1="{xa + dx/3:.2f}" y1="{ya + da*dx/3:.2f}" '
                   f'x2="{xb - dx/3:.2f}" y2="{yb - db*dx/3:.2f}" x3="{xb:.2f}" y3="{yb:.2f}"/>')
    return "".join(ops)


def stencil(name, w, h, paths, fill=False, dashed=(), filled_paths=()):
    fg = ""
    for p in filled_paths:
        fg += f"<path>{p}</path><fillstroke/>"
    for p in paths:
        fg += f"<path>{p}</path>" + ("<fillstroke/>" if fill else "<stroke/>")
    for p in dashed:
        fg += f'<dashed dashed="1"/><path>{p}</path><stroke/><dashed dashed="0"/>'
    xml = (f'<shape name="{name}" h="{h}" w="{w}" aspect="variable" strokewidth="inherit">'
           f'<connections/><background/><foreground>{fg}</foreground></shape>')
    return compress(xml)


# ---------- geometry helpers ----------
def rect(x, y, w, h):
    return f'<move x="{x:.2f}" y="{y:.2f}"/><line x="{x+w:.2f}" y="{y:.2f}"/><line x="{x+w:.2f}" y="{y+h:.2f}"/><line x="{x:.2f}" y="{y+h:.2f}"/><close/>'

def seg(x0, y0, x1, y1):
    return f'<move x="{x0:.2f}" y="{y0:.2f}"/><line x="{x1:.2f}" y="{y1:.2f}"/>'

def chevron(x, y, angle_deg, size=6):
    """Open arrowhead with its tip at (x, y), pointing along angle (0 = +x, 90 = +y)."""
    a = math.radians(angle_deg)
    out = []
    for s in (+1, -1):
        b = a + math.pi + s * math.radians(28)
        out.append(seg(x, y, x + size * math.cos(b), y + size * math.sin(b)))
    return "".join(out)

def arc_to(rx, ry, x, y, large=0, sweep=1):
    return f'<arc rx="{rx:.2f}" ry="{ry:.2f}" x-axis-rotation="0" large-arc-flag="{large}" sweep-flag="{sweep}" x="{x:.2f}" y="{y:.2f}"/>'

def polar(cx, cy, r, deg):
    return cx + r * math.cos(math.radians(deg)), cy + r * math.sin(math.radians(deg))

def ellipse_path(cx, cy, rx, ry):
    return (f'<move x="{cx-rx:.2f}" y="{cy:.2f}"/>' + arc_to(rx, ry, cx + rx, cy, 0, 1)
            + arc_to(rx, ry, cx - rx, cy, 0, 1) + "<close/>")

def cells(x, y, w, h, n, vertical=False):
    out = rect(x, y, w, h)
    for i in range(1, n):
        if vertical:
            out += seg(x, y + i * h / n, x + w, y + i * h / n)
        else:
            out += seg(x + i * w / n, y, x + i * w / n, y + h)
    return out

def grid(x, y, w, h, nx, ny):
    out = rect(x, y, w, h)
    for i in range(1, nx):
        out += seg(x + i * w / nx, y, x + i * w / nx, y + h)
    for j in range(1, ny):
        out += seg(x, y + j * h / ny, x + w, y + j * h / ny)
    return out


# ---------- signal functions (t in [0,1], y in px for a row of height h centred at mid) ----------
def gauss(t, sigma): return math.exp(-0.5 * ((t - 0.5) / sigma) ** 2)
def sine(f, mid=MID, amp=AMP):   return lambda t: mid - amp * math.sin(2 * math.pi * f * t)
def cosine(f, mid=MID, amp=AMP): return lambda t: mid - amp * math.cos(2 * math.pi * f * t)
def morlet(f, mid=MID, amp=AMP):
    s = MORLET_SIGMA_CYCLES / f
    return lambda t: mid - amp * gauss(t, s) * math.cos(2 * math.pi * f * (t - 0.5))
def morlet_env(f, sign, mid=MID, amp=AMP):
    s = MORLET_SIGMA_CYCLES / f
    return lambda t: mid - sign * amp * gauss(t, s)
def modulated(f, mid=MID, amp=AMP):
    return lambda t: mid - amp * gauss(t, WIDE_SIGMA) * math.cos(2 * math.pi * f * (t - 0.5))
def bell(sigma, mid=MID, amp=AMP, center=0.5):
    return lambda t: mid + amp - 2 * amp * math.exp(-0.5 * ((t - center) / sigma) ** 2)
def hann(mid=MID, amp=AMP): return lambda t: mid + amp - 2 * amp * 0.5 * (1 - math.cos(2 * math.pi * t))
def tukey(alpha=0.5):
    def w(t):
        if t < alpha / 2:     return 0.5 * (1 - math.cos(2 * math.pi * t / alpha))
        if t > 1 - alpha / 2: return 0.5 * (1 - math.cos(2 * math.pi * (1 - t) / alpha))
        return 1.0
    return lambda t: MID + AMP - 2 * AMP * w(t)
def butter_lp(fc, order=4): return lambda x: 1 / math.sqrt(1 + (x / fc) ** (2 * order))
def butter_hp(fc, order=4): return lambda x: 1 / math.sqrt(1 + (fc / max(x, 1e-6)) ** (2 * order))
def response(mag):          return lambda t: MID + AMP - 2 * AMP * mag(t)
def bandpass(center, bw, order=2):
    return lambda x: 1 / math.sqrt(1 + ((x - center) / (bw / 2)) ** (2 * order))
def closed(fn, n, base=MID + AMP, x0=0, x1=TILE_W):
    return hermite_path(fn, n, x0, x1) + f'<line x="{x1:.2f}" y="{base:.2f}"/><line x="{x0:.2f}" y="{base:.2f}"/><close/>'


# ---------- catalogue ----------
items = []

def add(group, key, title, paths, *, w=TILE_W, h=TILE_H, fill=False, dashed=(), filled=(), labels=(), fillcolor=None):
    items.append(dict(group=group, key=key, title=title, paths=list(paths), w=w, h=h, fill=fill,
                      dashed=list(dashed), filled=list(filled), labels=list(labels), fillcolor=fillcolor))

# --- signals ---
for f in FREQS:
    add("sines", f"sin{f}", f"sine {f} Hz", [hermite_path(sine(f), SEGS * f)])
for f in FREQS:
    add("cosines", f"cos{f}", f"cosine {f} Hz", [hermite_path(cosine(f), SEGS * f)])
for f in FREQS:
    x = TILE_W * f / 6
    add("spectrum", f"spike{f}", f"spectrum spike {f} Hz",
        [seg(0, MID + AMP, TILE_W, MID + AMP) + seg(x, MID + AMP, x, MID - AMP)])

for f in FREQS:
    n = SEGS * max(f, 2) * 2
    add("wavelets", f"morlet{f}", f"Morlet wavelet scale {f}", [hermite_path(morlet(f), n)])
for f in FREQS:
    n = SEGS * max(f, 2) * 2
    add("wavelets_env", f"morlet{f}_env", f"Morlet {f} + envelope", [hermite_path(morlet(f), n)],
        dashed=[hermite_path(morlet_env(f, +1), 32), hermite_path(morlet_env(f, -1), 32)])
for f in FREQS:
    n = SEGS * max(f, 2) * 2
    add("modulated", f"mod{f}", f"windowed tone {f} Hz", [hermite_path(modulated(f), n)],
        dashed=[hermite_path(lambda t: MID - AMP * gauss(t, WIDE_SIGMA), 32),
                hermite_path(lambda t: MID + AMP * gauss(t, WIDE_SIGMA), 32)])

windows = [("gauss_narrow", "Gaussian window (narrow)", bell(0.10)),
           ("gauss_wide", "Gaussian window (wide)", bell(0.20)),
           ("hann", "Hann window", hann()),
           ("tukey", "Tukey window", tukey(0.5))]
for key, title, fn in windows:
    add("windows", f"win_{key}", title, [hermite_path(fn, 32)])
    add("masks", f"mask_{key}", title + " (filled)", [closed(fn, 32)], fill=True)
top, bot = MID - AMP, MID + AMP
rect_win = f'<move x="0" y="{bot}"/><line x="24" y="{bot}"/><line x="24" y="{top}"/><line x="136" y="{top}"/><line x="136" y="{bot}"/><line x="{TILE_W}" y="{bot}"/>'
add("windows", "win_rect", "Rectangular window", [rect_win])
add("masks", "mask_rect", "Rectangular window (filled)", [rect_win + "<close/>"], fill=True)

def hann_chain(n_windows, filled=False):
    hop = 1 / (n_windows + 1)
    out = []
    for i in range(n_windows):
        a, b = i * hop, i * hop + 2 * hop
        p = hermite_path(hann(), 16, a * TILE_W, b * TILE_W)
        if filled:
            p += f'<line x="{b*TILE_W:.2f}" y="{bot:.2f}"/><line x="{a*TILE_W:.2f}" y="{bot:.2f}"/><close/>'
        out.append(p)
    return out
for n in (3, 5, 7):
    add("windows", f"win_hann_chain{n}", f"Hann chain x{n} (50% hop)", hann_chain(n))
    add("masks", f"mask_hann_chain{n}", f"Hann chain x{n} (filled)", hann_chain(n, True), fill=True)
for pct in (10, 25, 40):
    w = pct / 100 * TILE_W
    add("edge_masks", f"mask_edges{pct}", f"edge mask {pct}% each side",
        [rect(0, 0, w, TILE_H) + rect(TILE_W - w, 0, w, TILE_H)], fill=True)
add("edge_masks", "mask_left25", "left mask 25%", [rect(0, 0, 40, TILE_H)], fill=True)
add("edge_masks", "mask_full", "full mask", [rect(0, 0, TILE_W, TILE_H)], fill=True)

filters = [("lowpass", "lowpass", response(butter_lp(0.45))),
           ("highpass", "highpass", response(butter_hp(0.55))),
           ("bandpass", "bandpass", response(bandpass(0.5, 0.30))),
           ("bandpass_narrow", "bandpass (narrow)", response(bandpass(0.5, 0.16))),
           ("bandstop", "notch / bandstop", response(lambda x: 1 - bandpass(0.5, 0.20)(x)))]
for key, title, fn in filters:
    add("filters", f"filt_{key}", title, [hermite_path(fn, 48)])
    add("filters_filled", f"filt_{key}_fill", title + " (filled)", [closed(fn, 48)], fill=True)
def bank_paths(n_bands=5, filled=False):
    out = []
    for i in range(n_bands):
        c = (i + 0.5) / n_bands
        fn = response(lambda x, c=c: math.exp(-0.5 * ((x - c) / (0.425 / n_bands)) ** 2))
        out.append(closed(fn, 40) if filled else hermite_path(fn, 40))
    return out
add("filters", "filt_bank5", "bandpass bank (5)", bank_paths())
add("filters_filled", "filt_bank5_fill", "bandpass bank (5, filled)", bank_paths(filled=True), fill=True)

# --- hardware (160x80 tiles, glyph centred) ---
def cylinder(cx, top_y, rx, ry, height):
    body = (f'<move x="{cx-rx:.2f}" y="{top_y:.2f}"/><line x="{cx-rx:.2f}" y="{top_y+height:.2f}"/>'
            + arc_to(rx, ry, cx + rx, top_y + height, 0, 0)
            + f'<line x="{cx+rx:.2f}" y="{top_y:.2f}"/>')
    return body, ellipse_path(cx, top_y, rx, ry)
body, lid = cylinder(80, 16, 34, 9, 48)
add("hardware", "hw_disk", "disk", [body, lid])
add("hardware", "hw_disk_platter", "disk (platters)",
    [body, lid, ellipse_path(80, 16, 10, 2.6)] + [seg(46, 16 + k, 114, 16 + k) for k in (16, 32)])

file_body = '<move x="52" y="8"/><line x="96" y="8"/><line x="112" y="24"/><line x="112" y="72"/><line x="52" y="72"/><close/>'
file_fold = '<move x="96" y="8"/><line x="96" y="24"/><line x="112" y="24"/>'
add("hardware", "hw_audio_file", "audio file", [file_body, file_fold, hermite_path(sine(3, 46, 10), 24, 60, 104)])
add("hardware", "hw_file", "file", [file_body, file_fold])

speaker = ('<move x="46" y="30"/><line x="66" y="30"/><line x="88" y="14"/><line x="88" y="66"/><line x="66" y="50"/><line x="46" y="50"/><close/>'
           + seg(66, 30, 66, 50))
def arc_ring(cx, cy, r, a0, a1):
    x0, y0 = polar(cx, cy, r, a0); x1, y1 = polar(cx, cy, r, a1)
    return f'<move x="{x0:.2f}" y="{y0:.2f}"/>' + arc_to(r, r, x1, y1, 0, 1)
add("hardware", "hw_speaker", "sound device (out)", [speaker] + [arc_ring(88, 40, r, -35, 35) for r in (12, 22, 32)])
mic = (ellipse_path(80, 24, 9, 9) + seg(71, 24, 71, 40) + seg(89, 24, 89, 40) + arc_ring(80, 40, 9, 0, 180)
       + arc_ring(80, 40, 16, 0, 180) + seg(80, 56, 80, 66) + seg(68, 66, 92, 66))
add("hardware", "hw_mic", "sound device (in)", [mic])

monitor = (rect(30, 6, 100, 56) + rect(36, 11, 88, 46) + rect(72, 62, 16, 6) + seg(56, 72, 104, 72))
bars = "".join(seg(44 + i * 10, 52, 44 + i * 10, 52 - hgt) for i, hgt in enumerate((14, 30, 22, 36, 18, 28, 12, 24)))
add("hardware", "hw_display", "display", [monitor])
add("hardware", "hw_display_viz", "display (visualizing)", [monitor, bars])

def chip(cx, cy, w, h, pins_x, pins_y, pin_len=6, inner=None):
    out = rect(cx - w / 2, cy - h / 2, w, h)
    for i in range(pins_x):
        px = cx - w / 2 + (i + 0.5) * w / pins_x
        out += seg(px, cy - h / 2, px, cy - h / 2 - pin_len) + seg(px, cy + h / 2, px, cy + h / 2 + pin_len)
    for j in range(pins_y):
        py = cy - h / 2 + (j + 0.5) * h / pins_y
        out += seg(cx - w / 2, py, cx - w / 2 - pin_len, py) + seg(cx + w / 2, py, cx + w / 2 + pin_len, py)
    if inner:
        out += inner
    return out
add("hardware", "hw_cpu", "CPU", [chip(80, 40, 44, 44, 4, 4, inner=rect(68, 28, 24, 24))])
add("hardware", "hw_gpu", "GPU", [chip(80, 40, 88, 44, 8, 0, inner=grid(46, 26, 68, 28, 8, 2))])
ram = rect(20, 24, 120, 32) + "".join(rect(26 + i * 14, 30, 10, 20) for i in range(8)) + "".join(seg(24 + i * 6, 56, 24 + i * 6, 62) for i in range(19))
add("hardware", "hw_ram", "RAM / VRAM", [ram])
add("hardware", "hw_bus", "CPU | GPU divider", [seg(0, 38, 160, 38) + seg(0, 42, 160, 42)])

# square (100x100) hardware glyphs for unit tiles
body_sq, lid_sq = cylinder(50, 24, 34, 9, 50)
add("hardware", "hw_disk_sq", "disk (square)", [body_sq, lid_sq], w=100, h=100)
speaker_sq = ('<move x="16" y="38"/><line x="38" y="38"/><line x="60" y="20"/><line x="60" y="80"/><line x="38" y="62"/><line x="16" y="62"/><close/>' + seg(38, 38, 38, 62))
add("hardware", "hw_speaker_sq", "sound device (square)", [speaker_sq] + [arc_ring(60, 50, r, -35, 35) for r in (12, 22, 32)], w=100, h=100)
monitor_sq = rect(8, 16, 84, 56) + rect(14, 21, 72, 46) + rect(44, 72, 12, 6) + seg(28, 84, 72, 84)
add("hardware", "hw_display_sq", "display (square)", [monitor_sq], w=100, h=100)
bars_sq = "".join(seg(22 + i * 8, 62, 22 + i * 8, 62 - hgt) for i, hgt in enumerate((12, 26, 18, 32, 14, 24, 10, 20)))
add("hardware", "hw_display_viz_sq", "display visualizing (square)", [monitor_sq, bars_sq], w=100, h=100)

# --- arrays & buffers ---
for n in (4, 8, 16):
    add("arrays", f"array{n}", f"array x{n}", [cells(0, 20, 160, 40, n)], h=80)
add("arrays", "array8_head", "array x8 (write head)", [cells(0, 20, 160, 40, 8)], filled=[rect(60, 20, 20, 40)], fill=False, h=80)
wrap = seg(150, 60, 150, 72) + seg(150, 72, 10, 72) + seg(10, 72, 10, 60) + chevron(10, 60, -90)
add("arrays", "array8_wrap", "array x8 (wrap-around)", [cells(0, 20, 160, 40, 8) + wrap], h=80)
add("arrays", "array8_hop", "array x8 (hop + overlap)",
    [cells(0, 32, 160, 30, 8), seg(0, 22, 100, 22) + seg(0, 18, 0, 26) + seg(100, 18, 100, 26),
     seg(60, 10, 160, 10) + seg(60, 6, 60, 14) + seg(160, 6, 160, 14), seg(0, 70, 60, 70) + chevron(60, 70, 0) + seg(0, 66, 0, 74)], h=80)

def ring(cx, cy, r_out, r_in, n, gap_deg=4):
    out = []
    for i in range(n):
        a0 = -90 + i * 360 / n + gap_deg / 2
        a1 = -90 + (i + 1) * 360 / n - gap_deg / 2
        xo0, yo0 = polar(cx, cy, r_out, a0); xo1, yo1 = polar(cx, cy, r_out, a1)
        xi0, yi0 = polar(cx, cy, r_in, a0); xi1, yi1 = polar(cx, cy, r_in, a1)
        out.append(f'<move x="{xo0:.2f}" y="{yo0:.2f}"/>' + arc_to(r_out, r_out, xo1, yo1, 0, 1)
                   + f'<line x="{xi1:.2f}" y="{yi1:.2f}"/>' + arc_to(r_in, r_in, xi0, yi0, 0, 0) + "<close/>")
    return out
def ring_arrow(cx, cy, r, a0=-60, a1=200):
    x0, y0 = polar(cx, cy, r, a0); x1, y1 = polar(cx, cy, r, a1)
    tangent = a1 + 90
    return f'<move x="{x0:.2f}" y="{y0:.2f}"/>' + arc_to(r, r, x1, y1, 1 if a1 - a0 > 180 else 0, 1) + chevron(x1, y1, tangent, size=max(5, r * 0.3))
add("circular", "ring8", "ring buffer x8", ring(80, 40, 36, 22, 8) + [ring_arrow(80, 40, 12, -80, 190)])
add("circular", "ring12", "ring buffer x12", ring(80, 40, 36, 24, 12) + [ring_arrow(80, 40, 12, -80, 190)])
add("circular", "ring8_head", "ring buffer x8 (head)", ring(80, 40, 36, 22, 8), filled=[ring(80, 40, 36, 22, 8)[0]])
add("circular", "loop_arrow", "loop arrow", [ring_arrow(80, 40, 28, -60, 240)])
add("circular", "loop_arrow_ccw", "loop arrow (ccw)",
    [(lambda x0, y0, x1, y1: f'<move x="{x0:.2f}" y="{y0:.2f}"/>' + arc_to(28, 28, x1, y1, 1, 0) + chevron(x1, y1, 240 - 90 + 180 - 180, size=9))(*polar(80, 40, 28, 240), *polar(80, 40, 28, -60))])
slabs = "".join(rect(20, 8 + i * 12, 100, 9) for i in range(6))
loop = seg(132, 8 + 5 * 12 + 4.5, 146, 8 + 5 * 12 + 4.5) + seg(146, 8 + 5 * 12 + 4.5, 146, 12.5) + seg(146, 12.5, 130, 12.5) + chevron(130, 12.5, 180)
add("circular", "stack_loop", "stack + loop (circular buffer)", [slabs, loop])
add("circular", "stack_loop_head", "stack + loop (head)", [slabs, loop], filled=[rect(20, 8 + 2 * 12, 100, 9)])

def deck(n, w=96, h=60, dx=8, dy=6, x0=8, y0=None):
    if y0 is None:
        y0 = 8 + (n - 1) * dy
    return [rect(x0 + i * dx, y0 - i * dy, w, h) for i in range(n)]
add("stacks3d", "deck4", "frame deck x4", deck(4), fill=True, h=100)
add("stacks3d", "deck8", "frame deck x8", deck(8, w=88, h=56, dx=7, dy=5), fill=True, h=100)
def cube_slabs(n, x=16, y=28, w=88, h=56, d=40, ang=35):
    ox, oy = d * math.cos(math.radians(ang)), -d * math.sin(math.radians(ang))
    out = rect(x, y, w, h)
    out += f'<move x="{x:.2f}" y="{y:.2f}"/><line x="{x+ox:.2f}" y="{y+oy:.2f}"/><line x="{x+w+ox:.2f}" y="{y+oy:.2f}"/><line x="{x+w:.2f}" y="{y:.2f}"/>'
    out += f'<move x="{x+w+ox:.2f}" y="{y+oy:.2f}"/><line x="{x+w+ox:.2f}" y="{y+h+oy:.2f}"/><line x="{x+w:.2f}" y="{y+h:.2f}"/>'
    for i in range(1, n):
        fx, fy = ox * i / n, oy * i / n
        out += seg(x + fx, y + fy, x + w + fx, y + fy) + seg(x + w + fx, y + fy, x + w + fx, y + h + fy)
    return [out]
add("stacks3d", "cube_slabs6", "frame cube x6 (depth = time)", cube_slabs(6), h=100)
add("stacks3d", "cube_slabs12", "frame cube x12", cube_slabs(12), h=100)
add("stacks3d", "cube_slabs6_col", "frame cube x6 (newest slab)", cube_slabs(6),
    filled=[(lambda x, y, w, h, ox, oy: f'<move x="{x+ox*5/6:.2f}" y="{y+oy*5/6:.2f}"/><line x="{x+w+ox*5/6:.2f}" y="{y+oy*5/6:.2f}"/><line x="{x+w+ox:.2f}" y="{y+oy:.2f}"/><line x="{x+ox:.2f}" y="{y+oy:.2f}"/><close/>'
             f'<move x="{x+w+ox*5/6:.2f}" y="{y+oy*5/6:.2f}"/><line x="{x+w+ox:.2f}" y="{y+oy:.2f}"/><line x="{x+w+ox:.2f}" y="{y+h+oy:.2f}"/><line x="{x+w+ox*5/6:.2f}" y="{y+h+oy*5/6:.2f}"/><close/>')
            (16, 28, 88, 56, 40 * math.cos(math.radians(35)), -40 * math.sin(math.radians(35)))], h=100)

add("textures", "grid8x4", "texture 8x4", [grid(0, 0, 160, 80, 8, 4)])
add("textures", "grid16x8", "texture 16x8", [grid(0, 0, 160, 80, 16, 8)])
add("textures", "grid8x4_col", "texture 8x4 (new column)", [grid(0, 0, 160, 80, 8, 4)], filled=[rect(140, 0, 20, 80)])
add("textures", "grid8x4_scroll", "texture 8x4 (scrolling)", [grid(0, 0, 160, 80, 8, 4), seg(150, 88, 10, 88) + chevron(10, 88, 180)], filled=[rect(140, 0, 20, 80)], h=96)
add("textures", "double_buffer", "double buffer (swap)",
    [rect(8, 12, 56, 56), rect(96, 12, 56, 56), seg(68, 30, 92, 30) + chevron(92, 30, 0), seg(92, 50, 68, 50) + chevron(68, 50, 180)])
add("textures", "triple_buffer", "back / front / display",
    [rect(4, 12, 40, 56), rect(60, 12, 40, 56), rect(116, 12, 40, 56), seg(46, 40, 58, 40) + chevron(58, 40, 0), seg(102, 40, 114, 40) + chevron(114, 40, 0)])

# --- banks (5 stacked rows, 160x160; labels K0..K4 sit in a 40 px column on the left) ---
BANK_ROWS, ROW_H = 5, 32
def bank(kind):
    out = []
    for i in range(BANK_ROWS):
        y = i * ROW_H
        mid, amp = y + ROW_H / 2, ROW_H * 0.36
        f = BANK_ROWS - i        # highest scale on top, like the old diagram
        row = rect(0, y, 160, ROW_H)
        if kind == "time":
            row += hermite_path(modulated(f + 1, mid, amp), SEGS * (f + 1) * 2)
        elif kind == "morlet":
            row += hermite_path(morlet(f, mid, amp), SEGS * max(f, 2) * 2)
        elif kind == "freq":
            c = 0.12 + 0.76 * (f - 1) / (BANK_ROWS - 1)
            row += hermite_path(bell(0.045, mid, amp, c), 48)
        elif kind == "tone":
            row += hermite_path(sine(f, mid, amp), SEGS * f)
        elif kind == "coefs":
            c = 0.3 + 0.08 * f
            row += hermite_path(lambda t, c=c, f=f: mid + amp - 2 * amp * math.exp(-0.5 * ((t - c) / (0.14 - 0.012 * f)) ** 2), 48)
        elif kind == "empty":
            pass
        out.append(row)
    return out
bank_labels = [(f"K{BANK_ROWS - 1 - i}", 0, i * ROW_H, 36, ROW_H) for i in range(BANK_ROWS)]
for kind, title in [("time", "wavelet bank (time)"), ("morlet", "wavelet bank (dilated Morlet)"), ("freq", "wavelet bank (frequency)"),
                    ("tone", "tone bank"), ("coefs", "CWT coefficient stack"), ("empty", "labelled stack")]:
    add("banks", f"bank_{kind}", title, bank(kind), w=160, h=BANK_ROWS * ROW_H, labels=bank_labels)

# --- layered decks: cards drawn back to front, each card opaque so the front one wins ---
def deck_cards(kind, n=5, card_w=62, card_h=62, dx=7, dy=7, x0=2, y0=None, content="front"):
    """Card k=0 is the front (bottom-left); cards recede up-right. Drawn back to front."""
    if y0 is None:
        y0 = 2 + (n - 1) * dy
    filled, strokes = [], []
    for k in range(n - 1, -1, -1):
        x, y = x0 + k * dx, y0 - k * dy
        filled.append(rect(x, y, card_w, card_h))
        mid, amp = y + card_h / 2, card_h * 0.36
        f = k + 2
        if content == "front" and k > 0:
            strokes.append("")
        elif kind == "wavelets":
            strokes.append(hermite_path(modulated(f, mid, amp), SEGS * f * 2, x, x + card_w))
        elif kind == "morlet":
            strokes.append(hermite_path(morlet(f - 1, mid, amp), SEGS * max(f - 1, 2) * 2, x, x + card_w))
        elif kind == "freq":
            c = 0.15 + 0.7 * k / (n - 1)
            strokes.append(hermite_path(bell(0.05, mid, amp, c), 48, x, x + card_w))
        elif kind == "coefs":
            strokes.append(hermite_path(lambda t, k=k: mid + amp - 2 * amp * math.exp(-0.5 * ((t - 0.35 - 0.08 * k) / (0.13 - 0.012 * k)) ** 2), 48, x, x + card_w))
        elif kind == "tones":
            strokes.append(hermite_path(sine(f, mid, amp), SEGS * f, x, x + card_w))
        elif kind == "grid":
            strokes.append(grid(x, y, card_w, card_h, 8, 4))
        elif kind == "spikes":
            px = x + card_w * (0.15 + 0.7 * k / (n - 1))
            strokes.append(seg(x, y + card_h - 8, x + card_w, y + card_h - 8) + seg(px, y + card_h - 8, px, y + 8))
        else:
            strokes.append("")
    return filled, strokes

PAL_DSP, PAL_REND, PAL_AUDIO = (0x6A, 0x5C, 0xD6), (0xF0, 0x52, 0x1A), (0xE8, 0xA3, 0x17)
def ramp(v):
    """Fixed heat ramp, theme independent: black -> purple -> red -> orange -> yellow -> white."""
    stops = [(0.0, (0, 0, 0)), (0.2, PAL_DSP), (0.4, (0xD0, 0x22, 0x2A)), (0.6, PAL_REND),
             (0.8, (0xF5, 0xE0, 0x40)), (1.0, (255, 255, 255))]
    v = max(0.0, min(1.0, v))
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            t = (v - a) / (b - a)
            return "#%02X%02X%02X" % tuple(round(ca[i] + t * (cb[i] - ca[i])) for i in range(3))
    return "#%02X%02X%02X" % stops[-1][1]

def hue(v):
    """Heat hue without a white end: purple -> orange -> yellow. Intensity is carried by alpha (see `alpha`),
    so low values fade into whatever the canvas colour is - the same stencil works on a light or dark theme."""
    stops = [(0.0, PAL_DSP), (0.5, PAL_REND), (1.0, PAL_AUDIO)]
    v = max(0.0, min(1.0, v))
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            t = (v - a) / (b - a)
            return "#%02X%02X%02X" % tuple(round(ca[i] + t * (cb[i] - ca[i])) for i in range(3))
    return "#%02X%02X%02X" % stops[-1][1]

def alpha(v):
    return f"{max(0.0, min(1.0, v)) ** 0.5:.2f}"

def spectro(tx, fy):
    """Synthetic CWT-like intensity in [0,1]; tx = time 0..1, fy = 0 (low freq, bottom) .. 1 (high, top)."""
    g = lambda a, s: math.exp(-0.5 * (a / s) ** 2)
    tone = g(fy - 0.3, 0.06) * (0.55 + 0.45 * math.sin(2 * math.pi * 1.5 * tx))
    chirp = g(fy - (0.15 + 0.75 * tx), 0.06) * 0.9
    burst = g(tx - 0.72, 0.07) * g(fy - 0.72, 0.16)
    return min(1.0, 0.95 * max(tone, chirp, burst))

def weave(tx, fy, cycles=4.0, contrast=1):
    """dsplot's Gramian Angular Field weave (research/dsplot/figures/sample_template.py): diagonal-band
    interference of one sine against itself in [0,1]. `contrast` smoothstep passes push peaks up and troughs down
    while keeping the bands."""
    phi = lambda t: math.acos(max(-1.0, min(1.0, math.sin(2 * math.pi * cycles * t))))
    v = (math.cos(phi(tx) + phi(fy)) + 1.0) / 2.0
    for _ in range(contrast):
        v = v * v * (3 - 2 * v)
    return v

def heatmap_ops(x, y, w, h, nx, ny, outline=True, mode="color", field=None):
    """mode="color": opaque cells through the fixed ramp (a shader output, theme independent).
    mode="ink": cells are thick strokes in the shape's own stroke colour with alpha = intensity, so raw
    intensity is black-on-light and white-on-dark. Canvas state is saved/restored around the cells."""
    cw, ch = w / nx, h / ny
    ops = "<save/>"
    if mode == "ink":
        ops += f'<linecap cap="flat"/><strokewidth width="{ch:.2f}"/>'
    for j in range(ny):
        for i in range(nx):
            v = (field or spectro)((i + 0.5) / nx, 1 - (j + 0.5) / ny)
            if mode == "ink" and v < 0.03:
                continue
            cx, cy = x + i * cw, y + j * ch
            if mode == "ink":
                ops += f'<strokealpha alpha="{alpha(v)}"/><path><move x="{cx:.2f}" y="{cy + ch / 2:.2f}"/><line x="{cx + cw:.2f}" y="{cy + ch / 2:.2f}"/></path><stroke/>'
            else:
                ops += f'<fillcolor color="{ramp(v)}"/><rect x="{cx:.2f}" y="{cy:.2f}" w="{cw:.2f}" h="{ch:.2f}"/><fill/>'
    ops += "<restore/>"
    if outline:
        ops += f"<path>{rect(x, y, w, h)}</path><stroke/>"
    return ops

def checker_ops(x, y, w, h, n, outline=True, color="#000000", other=None):
    """Checkerboard: `color` on the odd cells; `other` on the even cells, or the canvas when None. Painted
    black-and-white it is a fixed test pattern, theme independent: an allocated, empty frame."""
    ops = "<save/>"
    for j in range(n):
        for i in range(n):
            c = color if (i + j) % 2 else other
            if c is None:
                continue
            ops += f'<fillcolor color="{c}"/><rect x="{x + i * w / n:.2f}" y="{y + j * h / n:.2f}" w="{w / n:.2f}" h="{h / n:.2f}"/><fill/>'
    ops += "<restore/>"
    if outline:
        ops += f"<path>{rect(x, y, w, h)}</path><stroke/>"
    return ops

def cmap_ops(x, y, w, h, n, vertical=True):
    """Colour map stripes: the fixed opaque ramp (black .. white), identical on light and dark themes."""
    ops = "<save/>"
    for k in range(n):
        v = 1 - (k + 0.5) / n if vertical else (k + 0.5) / n
        ops += f'<fillcolor color="{ramp(v)}"/>'
        if vertical:
            ops += f'<rect x="{x:.2f}" y="{y + k * h / n:.2f}" w="{w:.2f}" h="{h / n:.2f}"/><fill/>'
        else:
            ops += f'<rect x="{x + k * w / n:.2f}" y="{y:.2f}" w="{w / n:.2f}" h="{h:.2f}"/><fill/>'
    ops += f"<restore/><path>{rect(x, y, w, h)}</path><stroke/>"
    return ops

RAW = {}
def stencil_raw(name, w, h, fg):
    xml = (f'<shape name="{name}" h="{h}" w="{w}" aspect="variable" strokewidth="inherit">'
           f'<connections/><background/><foreground>{fg}</foreground></shape>')
    return compress(xml)

def add_raw(group, key, title, fg, w=TILE_W, h=TILE_H, fillcolor=None):
    RAW[key] = fg
    add(group, key, title, [], w=w, h=h, fillcolor=fillcolor)

def deck_fg(kind, n=9, card=50, d=50 / 8, x0=0, extra=""):
    """Opaque cards in the shape's own fill (theme canvas colour), back to front; every card carries its own content.
    Each card is one square unit (50) and the 9-card stack is one unit deep, filling a 100 x 100 tile."""
    y0 = (n - 1) * d
    fg = ""
    for k in range(n - 1, -1, -1):
        x, y = x0 + k * d, y0 - k * d
        fg += f'<path>{rect(x, y, card, card)}</path><fillstroke/>'
        mid, amp = y + card / 2, card * 0.36
        f = k + 2
        if kind == "wavelets":
            fg += f"<path>{hermite_path(modulated(f, mid, amp), SEGS * f * 2, x, x + card)}</path><stroke/>"
        elif kind == "morlet":
            fg += f"<path>{hermite_path(morlet(f - 1, mid, amp), SEGS * max(f - 1, 2) * 2, x, x + card)}</path><stroke/>"
        elif kind == "freq":
            c = 0.15 + 0.7 * k / (n - 1)
            fg += f"<path>{hermite_path(bell(0.05, mid, amp, c), 48, x, x + card)}</path><stroke/>"
        elif kind == "coefs":
            fg += f"<path>{hermite_path(lambda t, k=k: mid + amp - 2 * amp * math.exp(-0.5 * ((t - 0.35 - 0.08 * k) / (0.13 - 0.012 * k)) ** 2), 48, x, x + card)}</path><stroke/>"
        elif kind == "tones":
            fg += f"<path>{hermite_path(sine(f, mid, amp), SEGS * f, x, x + card)}</path><stroke/>"
        elif kind == "grid":
            fg += f"<path>{grid(x, y, card, card, 8, 4)}</path><stroke/>"
        elif kind == "heatmap":
            fg += heatmap_ops(x, y, card, card, 8, 6, outline=True)
        elif kind == "heatmap_gray":
            fg += heatmap_ops(x, y, card, card, 8, 6, outline=True, mode="ink")
        elif kind == "checker":
            fg += checker_ops(x, y, card, card, 5, color="#000000", other="#FFFFFF")
        elif kind == "spikes":
            px = x + card * (0.15 + 0.7 * k / (n - 1))
            fg += f"<path>{seg(x, y + card - 8, x + card, y + card - 8) + seg(px, y + card - 8, px, y + 8)}</path><stroke/>"
    if extra:
        fg += f"<path>{extra}</path><stroke/>"
    return fg

deck_loop = (seg(60, 99, 99, 99) + seg(99, 99, 99, 1) + seg(99, 1, 94, 1) + chevron(94, 1, 180, size=5))
for key, title, kind, extra in [
        ("deck_frames", "frame deck", "frames", ""),
        ("deck_frames_loop", "frame deck + loop (circular)", "frames", deck_loop),
        ("deck_wavelets", "Wavelet Bank [T]", "wavelets", ""),
        ("deck_morlet", "Wavelet Bank [T] (dilated Morlet)", "morlet", ""),
        ("deck_freq", "Wavelet Bank [F]", "freq", ""),
        ("deck_coefs", "CWT coefficient deck", "coefs", ""),
        ("deck_tones", "tone deck", "tones", ""),
        ("deck_spikes", "spectrum deck", "spikes", ""),
        ("deck_grids", "texture deck", "grid", ""),
        ("deck_heatmap", "heatmap deck (frame buffer)", "heatmap", ""),
        ("deck_heatmap_gray", "heatmap deck, grayscale (frame buffer, raw intensity)", "heatmap_gray", ""),
        ("deck_checker", "checkerboard deck (allocated, empty)", "checker", ""),
        ("deck_heatmap_loop", "heatmap deck + loop (circular)", "heatmap", deck_loop)]:
    add_raw("decks", key, title, deck_fg(kind, extra=extra), w=100, h=100, fillcolor="default")

add_raw("heatmaps", "heatmap", "heatmap (wide)", heatmap_ops(0, 0, 160, 80, 16, 8), w=160, h=80)
add_raw("heatmaps", "heatmap_square", "heatmap (square)", heatmap_ops(0, 0, 100, 100, 10, 10), w=100, h=100)
add_raw("heatmaps", "heatmap_square_gray", "heatmap, intensity ink (square)", heatmap_ops(0, 0, 100, 100, 10, 10, mode="ink"), w=100, h=100)
add_raw("heatmaps", "heatmap_square_weave_gray", "weave (GAF) test pattern, intensity ink (square)", heatmap_ops(0, 0, 100, 100, 32, 32, mode="ink", field=lambda tx, fy: weave(tx, fy, 3.0)), w=100, h=100)
add_raw("heatmaps", "heatmap_square_gray_col", "heatmap, intensity ink + new column (square)", heatmap_ops(0, 0, 100, 100, 10, 10, mode="ink") + f"<path>{rect(90, 0, 10, 100)}</path><stroke/>", w=100, h=100)
add_raw("heatmaps", "heatmap_gray", "heatmap, grayscale (wide)", heatmap_ops(0, 0, 160, 80, 16, 8, mode="ink"), w=160, h=80)
add_raw("heatmaps", "heatmap_gray_col", "heatmap, grayscale + new column", heatmap_ops(0, 0, 160, 80, 16, 8, mode="ink") + f"<path>{rect(150, 0, 10, 80)}</path><stroke/>", w=160, h=80)
add_raw("heatmaps", "heatmap_col", "heatmap + new column", heatmap_ops(0, 0, 160, 80, 16, 8) + f"<path>{rect(150, 0, 10, 80)}</path><stroke/>", w=160, h=80)
add_raw("heatmaps", "cmap_vertical", "color map (vertical)", cmap_ops(0, 0, 40, 100, 16, True), w=40, h=100)
add_raw("heatmaps", "cmap_square", "color map (square)", cmap_ops(0, 0, 100, 100, 16, True), w=100, h=100)
add_raw("heatmaps", "cmap_horizontal", "color map (horizontal)", cmap_ops(0, 0, 160, 40, 16, False), w=160, h=40)

# --- windowing scheme ---
def strip_with_windows(n_cells, n_windows, overlap):
    strip = cells(0, 56, 160, 16, n_cells)
    win_w = 160 / (1 + (n_windows - 1) * (1 - overlap))
    hop = win_w * (1 - overlap)
    brackets = []
    for i in range(n_windows):
        x0 = i * hop
        y = 44 - i * 12
        brackets.append(seg(x0, y, x0 + win_w, y) + seg(x0, y - 4, x0, y + 4) + seg(x0 + win_w, y - 4, x0 + win_w, y + 4))
    return [strip] + brackets
add("scheme", "scheme_overlap50", "chunks: 50% overlap", strip_with_windows(16, 3, 0.5))
add("scheme", "scheme_overlap75", "chunks: 75% overlap", strip_with_windows(16, 4, 0.75))
add("scheme", "scheme_nooverlap", "chunks: no overlap", strip_with_windows(16, 3, 0.0))
def strip_hann(n_windows, overlap):
    strip = cells(0, 56, 160, 16, 16)
    win_w = 160 / (1 + (n_windows - 1) * (1 - overlap))
    hop = win_w * (1 - overlap)
    return [strip] + [hermite_path(hann(28, 24), 16, i * hop, i * hop + win_w) for i in range(n_windows)]
add("scheme", "scheme_hann50", "chunks + Hann (50%)", strip_hann(3, 0.5))
add("scheme", "scheme_hann75", "chunks + Hann (75%)", strip_hann(5, 0.75))
add("scheme", "scheme_hop", "hop glyph", [rect(10, 20, 80, 40), rect(70, 20, 80, 40), seg(10, 72, 70, 72) + chevron(70, 72, 0) + seg(10, 68, 10, 76)])

# --- flow badges ---
add("flow", "xfer_right", "transfer (right)", [rect(0, 0, 80, 40), seg(16, 20, 64, 20) + chevron(64, 20, 0)], w=80, h=40)
add("flow", "xfer_left", "transfer (left)", [rect(0, 0, 80, 40), seg(64, 20, 16, 20) + chevron(16, 20, 180)], w=80, h=40)
add("flow", "xfer_both", "transfer (both)", [rect(0, 0, 80, 40), seg(16, 14, 64, 14) + chevron(64, 14, 0), seg(64, 26, 16, 26) + chevron(16, 26, 180)], w=80, h=40)
add("flow", "xfer_down", "transfer (down)", [rect(0, 0, 40, 80), seg(20, 16, 20, 64) + chevron(20, 64, 90)], w=40, h=80)
add("flow", "xfer_up", "transfer (up)", [rect(0, 0, 40, 80), seg(20, 64, 20, 16) + chevron(20, 16, -90)], w=40, h=80)
add("flow", "process", "process", [rect(0, 0, 80, 40)], w=80, h=40)
add("flow", "process_round", "process (rounded)", ['<roundrect x="0" y="0" w="80" h="40" arcsize="12"/>'], w=80, h=40)

# ---------- emit ----------
sections = [
    ("Sinusoids", ["sines", "cosines", "spectrum"]),
    ("Wavelets", ["wavelets", "wavelets_env", "modulated"]),
    ("Windows", ["windows", "masks", "edge_masks"]),
    ("Filters", ["filters", "filters_filled"]),
    ("Hardware", ["hardware"]),
    ("Arrays & Buffers", ["arrays", "circular", "stacks3d", "textures"]),
    ("Decks (layered)", ["decks"]),
    ("Heatmaps & Color Maps", ["heatmaps"]),
    ("Banks (rows)", ["banks"]),
    ("Windowing Scheme", ["scheme"]),
    ("Flow", ["flow"]),
]

def item_style(it, st):
    if it.get("fillcolor"):
        return f"shape=stencil({st});{STROKE}fillColor={it['fillcolor']};"
    return f"shape=stencil({st});{STROKE}" + (FILL if (it["fill"] or it["filled"]) else "fillColor=none;")

def tile_cells(idx, it, st, x, y):
    w, h = it["w"], it["h"]
    label_w = 40 if it["labels"] else 0
    gid = f"t{idx}"
    out = f'''
        <mxCell id="{gid}" value="" style="group" vertex="1" connectable="0" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{w + label_w}" height="{h}" as="geometry" />
        </mxCell>
        <mxCell id="{gid}b" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#000000;" vertex="1" parent="{gid}">
          <mxGeometry x="{label_w}" width="{w}" height="{h}" as="geometry" />
        </mxCell>
        <mxCell id="{gid}s" value="" style="{item_style(it, st)}" vertex="1" parent="{gid}">
          <mxGeometry x="{label_w}" width="{w}" height="{h}" as="geometry" />
        </mxCell>'''
    for j, (text, lx, ly, lw, lh) in enumerate(it["labels"]):
        out += f'''
        <mxCell id="{gid}l{j}" value="{escape(text)}" style="text;html=1;align=right;verticalAlign=middle;fontSize=11;fontColor=#000000;" vertex="1" parent="{gid}">
          <mxGeometry x="{lx}" y="{ly}" width="{lw}" height="{lh}" as="geometry" />
        </mxCell>'''
    out += f'''
        <mxCell id="{gid}c" value="{escape(it["title"])}" style="text;html=1;align=center;verticalAlign=top;fontSize=11;fontColor=#666666;" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y + h + 4}" width="{w + label_w}" height="20" as="geometry" />
        </mxCell>'''
    return out

stencils = {it["key"]: (stencil_raw(it["key"], it["w"], it["h"], RAW[it["key"]]) if it["key"] in RAW
                        else stencil(it["key"], it["w"], it["h"], it["paths"], it["fill"], it["dashed"], it["filled"])) for it in items}


def main():
    cells_xml = ""
    idx = 0
    y = ORIGIN_Y
    for header, groups in sections:
        cells_xml += f'''
            <mxCell id="hdr-{re.sub(r"[^a-z]", "", header.lower())}" value="{escape(header)}" style="text;html=1;align=left;verticalAlign=middle;fontSize=16;fontStyle=1;fontColor=#000000;" vertex="1" parent="1">
              <mxGeometry x="{ORIGIN_X}" y="{y}" width="300" height="{HEADER_H}" as="geometry" />
            </mxCell>'''
        y += HEADER_H
        for group in groups:
            group_items = [it for it in items if it["group"] == group]
            x = ORIGIN_X
            row_h = max(it["h"] for it in group_items)
            for it in group_items:
                cells_xml += tile_cells(idx, it, stencils[it["key"]], x, y)
                x += it["w"] + (40 if it["labels"] else 0) + 40
                idx += 1
            y += row_h + ROW_GAP
        y += ROW_GAP

    diagram = f'''  <diagram name="Primitives" id="prim-all">
        <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1900" pageHeight="{y + ORIGIN_Y}" math="0" shadow="0">
          <root>
            <mxCell id="0" />
            <mxCell id="1" parent="0" />{cells_xml}
          </root>
        </mxGraphModel>
      </diagram>'''

    library = []
    for it in items:
        model = (f'<mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                 f'<mxCell id="2" value="" style="{item_style(it, stencils[it["key"]])}" vertex="1" parent="1">'
                 f'<mxGeometry width="{it["w"]}" height="{it["h"]}" as="geometry"/></mxCell></root></mxGraphModel>')
        library.append({"xml": compress(model), "w": it["w"], "h": it["h"], "title": it["title"], "aspect": "variable"})

    src_path, out_path, lib_path = sys.argv[1], sys.argv[2], sys.argv[3]
    src = open(src_path).read()
    src = re.sub(r'\n  <diagram name="(Primitives|Sinusoids|Wavelets|Windows|Filters)".*?</diagram>', "", src, flags=re.S)
    open(out_path, "w").write(src.replace("</mxfile>", diagram + "\n</mxfile>"))
    open(lib_path, "w").write("<mxlibrary>" + json.dumps(library) + "</mxlibrary>")
    print(f"{len(items)} primitives, {len(sections)} sections")


if __name__ == "__main__":
    main()
