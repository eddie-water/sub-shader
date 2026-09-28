"""Generate the README modules flowchart (the SubShader container around Audio File -> Audio -> DSP -> Render -> Display,
with the Sound Device above Audio) as a draw.io file.

usage: python research/tools/drawio_modules.py <out.drawio>

Replaces the hand-drawn assets/timing/subshader_modules.drawio as the export source. Frame, line and label sizes
come from the README display spec in style, converted for this page's width and its 500 px README slot.
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
sys.path.insert(0, __file__.rsplit("/", 3)[0])
from research.dsplot import style
from drawio_pipeline import AUDIO, DSP, REND, FONT_FAMILY, INK, Page

SHOWN_PX = 500                       # README width="500"
BOX, GAP, PAD = 100, 40, 40          # square module box, arrow run between boxes, container inset
X0, Y0 = 80, 80                      # container corner

ROW = [("Audio\nFile", INK), ("Audio", AUDIO), ("DSP", DSP), ("Render", REND), ("Display", INK)]


def modules_page():
    pg = Page("Modules", 0)
    width = 2 * PAD + len(ROW) * BOX + (len(ROW) - 1) * GAP
    height = 2 * PAD + 2 * BOX + GAP
    W_FRAME, W_BOX, FONT = (round(style.readme_drawio(px, width, SHOWN_PX), 1)
                            for px in (style.README_FRAME_PX, style.README_LINE_PX, style.README_TEXT_PX))
    pg.vertex(f"rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor={INK};strokeWidth={W_FRAME};"
              f"fontColor={INK};fontFamily={FONT_FAMILY};fontSize={FONT};fontStyle=1;verticalAlign=top;align=left;"
              f"spacing=0;spacingLeft=10;spacingTop=1;labelBackgroundColor=none;", X0, Y0, width, height, "SubShader")
    y_top, y_row = Y0 + PAD, Y0 + PAD + BOX + GAP
    ids = [pg.box(label, X0 + PAD + i * (BOX + GAP), y_row, color, w=BOX, h=BOX, stroke=W_BOX, size=FONT)
           for i, (label, color) in enumerate(ROW)]
    device = pg.box("Sound\nDevice", X0 + PAD + BOX + GAP, y_top, INK, w=BOX, h=BOX, stroke=W_BOX, size=FONT)
    for src, dst in zip(ids, ids[1:]):
        pg.edge(src, dst, width=W_BOX, exit_=(1, 0.5), entry=(0, 0.5))
    pg.edge(ids[1], device, width=W_BOX, exit_=(0.5, 0), entry=(0.5, 1))
    return pg.xml(Y0 + height + 40)


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        f.write(f'<mxfile host="drawio_modules.py">\n{modules_page()}\n</mxfile>')
    print("wrote", sys.argv[1])
