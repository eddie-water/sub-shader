"""Export one page of a .drawio file as the README's black-background PNG.

usage: python tools/drawio_export.py <file.drawio> <page index or name> <out.png> [scale]

The generated pages leave colours as ``default`` so they follow the draw.io
theme; here they are pinned to ``style.NEUTRAL_COLOR`` on ``style.BG_COLOR``,
rendered with the draw.io desktop CLI (Windows binary via WSL interop),
composited onto the background and cropped to the shared README edge pad.
"""
import os
import re
import subprocess
import sys

from PIL import Image

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from research.dsplot import style
from research.dsplot.export import crop_to_content

DRAWIO_EXE = "/mnt/c/Program Files/draw.io/draw.io.exe"
WIN_TMP = r"C:\Users\edevl\AppData\Local\Temp\subshader_drawio"
WSL_TMP = "/mnt/c/Users/edevl/AppData/Local/Temp/subshader_drawio"
SCALE = 1.5


def isolate_page(xml, page):
    diagrams = re.findall(r"\s*<diagram .*?</diagram>", xml, flags=re.S)
    if not page.isdigit():
        page = next(i for i, d in enumerate(diagrams) if f'name="{page}"' in d)
    return f'<mxfile host="drawio_export.py">{diagrams[int(page)]}\n</mxfile>'


def pin_theme(xml, ink=style.NEUTRAL_COLOR, bg=style.BG_COLOR):
    for key, val in (("strokeColor", ink), ("fontColor", ink), ("fillColor", bg), ("labelBackgroundColor", bg)):
        xml = xml.replace(f"{key}=default", f"{key}={val}")
    return xml.replace("light-dark(#000000,#FFFFFF)", ink)


def export(src, page, out, scale=SCALE):
    os.makedirs(WSL_TMP, exist_ok=True)
    with open(f"{WSL_TMP}/x.drawio", "w") as f:
        f.write(pin_theme(isolate_page(open(src).read(), page)))
    subprocess.run([DRAWIO_EXE, "-x", "-f", "png", "-s", str(scale), "-b", "0", "-t",
                    "-o", rf"{WIN_TMP}\x.png", rf"{WIN_TMP}\x.drawio"], check=True)
    im = Image.open(f"{WSL_TMP}/x.png").convert("RGBA")
    bg_rgb = tuple(int(style.BG_COLOR.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    canvas = Image.new("RGBA", im.size, bg_rgb + (255,))
    canvas.alpha_composite(im)
    canvas.convert("RGB").save(out)
    crop_to_content(out)
    print(out, Image.open(out).size)
    return out


if __name__ == "__main__":
    src, page, out = sys.argv[1:4]
    export(src, page, out, float(sys.argv[4]) if len(sys.argv) > 4 else SCALE)
