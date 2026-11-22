"""Compile the TikZ/PGFPlots figures (pdflatex) and export them to vector SVG and 600-dpi PNG.

Usage: python src/export_figures.py [--figures figures] [--no-compile]
"""
import argparse
import os
import shutil
import subprocess

import fitz  # PyMuPDF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = ["fig1_architecture", "fig2_realdata", "fig3_results"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--no-compile", action="store_true", help="only export existing PDFs")
    a = ap.parse_args()
    if not a.no_compile:
        if shutil.which("pdflatex") is None:
            raise SystemExit("pdflatex not found; install TeX Live or MiKTeX, or use --no-compile")
        for f in FIGS:
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f + ".tex"],
                           cwd=a.figures, check=True, stdout=subprocess.DEVNULL)
    for f in FIGS:
        pg = fitz.open(os.path.join(a.figures, f + ".pdf"))[0]
        svg = pg.get_svg_image(matrix=fitz.Identity, text_as_path=True)
        with open(os.path.join(a.figures, f + ".svg"), "w", encoding="utf-8") as fh:
            fh.write(svg)
        pg.get_pixmap(dpi=600, alpha=False).save(os.path.join(a.figures, f + ".png"))
        r = pg.rect
        print(f"{f}: {r.width / 72 * 2.54:.2f} x {r.height / 72 * 2.54:.2f} cm")


if __name__ == "__main__":
    main()
