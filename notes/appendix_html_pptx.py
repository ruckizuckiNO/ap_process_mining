"""Convert the 14 HTML reports into one appendix deck: output/appendix_html_reports.pptx.

Each report is rendered in headless Chromium (Playwright), all <details> opened,
and cut into 16:9 slide images at blank rows, preferring cuts just above an <h2>.
Run: conda run -n datasci python notes/appendix_html_pptx.py
Needs playwright-python (conda-forge) and `playwright install chromium`.
"""
import io
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "appendix_html_reports.pptx"
REPO = "https://github.com/ruckizuckiNO/ap_process_mining/blob/main/"

SECTIONS = [
    ("A.1 Definitions and clocks", [
        "docs/assumptions.html",
        "output/report_assumptions_ctrl1.html",
        "output/report_invoice_clock.html",
        "output/report_zjd1_znb1_clock.html",
        "output/report_znb1_end_wait_gr.html",
    ]),
    ("A.2 Process and routing", [
        "output/report.html",
        "output/report_znb1.html",
        "output/report_ctrl1_direct.html",
        "output/report_znb1_ctrl1.html",
        "output/report_znb1_ctrl1_model.html",
    ]),
    ("A.3 Queue, targets and control", [
        "output/report_queue_fifo.html",
        "output/report_po_cut_gbr.html",
        "output/report_scenario_po_cut.html",
        "output/report_control_charts.html",
    ]),
]

NAVY = RGBColor(0x14, 0x2B, 0x45)
TEAL = RGBColor(0x1F, 0x9E, 0xA3)
LIME = RGBColor(0xA3, 0xC9, 0x3A)
GREY = RGBColor(0x5B, 0x67, 0x76)
PALE = RGBColor(0xC9, 0xDD, 0xE8)
CARD = RGBColor(0xE8, 0xF3, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"

SCALE = 2            # device pixels per CSS pixel
PAD = 20             # CSS px kept left and right of the content column
# Image area on the slide (inches): below a slim header, above the footer
IMG_X, IMG_Y, IMG_W, IMG_H = 0.3, 0.62, 12.733, 6.55
MAX_STRETCH = 1.2    # a chunk may be this much taller than ideal (then shrinks)


def render(page, path, keep=None):
    """Return (title, full-height PIL image of the content column, h2 tops in CSS px).

    keep: optional list of (first, last) index ranges of the content column's
    children to show (inclusive; last=None runs to the end). Others are hidden.
    """
    page.goto(path.resolve().as_uri())
    page.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
    if keep is not None:
        page.evaluate("""(keep) => {
            const m = document.querySelector('main, .wrap') || document.body;
            const kids = [...m.children];
            kids.forEach((c, i) => {
                const ok = keep.some(([a, b]) => i >= a && (b === null || i <= b));
                if (!ok) c.style.display = 'none';
            });
        }""", [list(r) for r in keep])
    page.wait_for_timeout(400)
    info = page.evaluate("""() => {
        const m = document.querySelector('main, .wrap') || document.body;
        const r = m.getBoundingClientRect();
        const h2 = [...m.querySelectorAll('h2')].map(e => e.getBoundingClientRect().top + scrollY);
        return {x: r.x + scrollX, y: r.y + scrollY, w: r.width,
                h: document.documentElement.scrollHeight, h2: h2, title: document.title};
    }""")
    x0 = max(0, info["x"] - PAD)
    w = min(info["w"] + 2 * PAD, 1280 - x0)
    total, seg, parts = info["h"], 5000, []
    for y in range(0, total, seg):
        h = min(seg, total - y)
        png = page.screenshot(clip=dict(x=x0, y=y, width=w, height=h), full_page=True)
        parts.append(Image.open(io.BytesIO(png)).convert("RGB"))
    full = Image.new("RGB", (parts[0].width, sum(p.height for p in parts)), "white")
    yy = 0
    for p in parts:
        full.paste(p, (0, yy))
        yy += p.height
    return info["title"], full, [t * SCALE for t in info["h2"]]


def trim_bottom(img):
    """Drop trailing blank rows."""
    a = np.asarray(img.convert("L"), dtype=np.int16)
    spread = a.max(axis=1) - a.min(axis=1)
    rows = np.nonzero(spread > 6)[0]
    return img.crop((0, 0, img.width, int(rows[-1]) + 20 * SCALE)) if len(rows) else img


def cut(img, h2_tops):
    """Split the tall image into chunks that fit the slide image area."""
    ideal = int(img.width * IMG_H / IMG_W)
    a = np.asarray(img.convert("L"), dtype=np.int16)
    blank = (a.max(axis=1) - a.min(axis=1)) <= 6
    light = a.min(axis=1) > 185
    # cut points = middles of blank runs at least MIN_GAP tall (paragraph gaps,
    # not the gaps between lines of one paragraph)
    MIN_GAP = 11 * SCALE
    edges = np.diff(np.concatenate([[0], blank.astype(np.int8), [0]]))
    run_s, run_e = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    keep = (run_e - run_s) >= MIN_GAP
    mids = ((run_s + run_e) // 2)[keep]
    ends = run_e[keep]
    chunks, start, H = [], 0, img.height
    while H - start > ideal * MAX_STRETCH:
        hi, hmax = start + ideal, start + int(ideal * MAX_STRETCH)
        c = None
        # 1) the gap just above a section heading, if the slide is at least 2/3 full
        for t in sorted((t for t in h2_tops if start + 0.66 * ideal <= t <= hi), reverse=True):
            near = mids[(ends <= t + 2) & (ends >= t - 60 * SCALE)]
            if len(near):
                c = int(near[-1])
                break
        # 2) the lowest paragraph gap within the ideal height, else within the stretch
        if c is None:
            for a_, b_ in ((start + int(0.5 * ideal), hi), (hi, hmax)):
                m = mids[(mids > a_) & (mids <= b_)]
                if len(m):
                    c = int(m[-1])
                    break
        # 3) a text-free gap inside a tall box (no dark pixels in the row)
        if c is None:
            seg = light[start + int(0.5 * ideal):hi]
            e = np.diff(np.concatenate([[0], seg.astype(np.int8), [0]]))
            s_, e_ = np.nonzero(e == 1)[0], np.nonzero(e == -1)[0]
            ok = (e_ - s_) >= 12 * SCALE
            if ok.any():
                c = start + int(0.5 * ideal) + int(((s_ + e_) // 2)[ok][-1])
        if c is None:
            c = hi  # forced cut (a block taller than a slide)
        chunks.append(img.crop((0, start, img.width, c)))
        start = c
    chunks.append(img.crop((0, start, img.width, H)))
    return chunks


def tb(slide, x, y, w, h, text, size, color, bold=False, align=PP_ALIGN.LEFT, link=None,
       anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paras = text if isinstance(text, list) else [text]
    for i, t in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = t
        r.font.name, r.font.size, r.font.bold = FONT, Pt(size), bold
        r.font.color.rgb = color
        if link:
            r.hyperlink.address = link
    return box


def rect(slide, x, y, w, h, fill):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def main():
    rendered = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=dict(width=1280, height=900), device_scale_factor=SCALE)
        for sec, files in SECTIONS:
            for f in files:
                title, img, h2 = render(page, ROOT / f)
                chunks = cut(trim_bottom(img), h2)
                rendered.append((sec, f, title, chunks))
                print(f"{f}: {len(chunks)} slides")
        browser.close()

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    # title slide
    s = prs.slides.add_slide(blank)
    rect(s, 0, 0, 13.333, 7.5, NAVY)
    tb(s, 0.9, 1.0, 11, 0.4, "INN354 / SYSTEMS AND PROCESS INNOVATION · APPENDIX", 14, LIME, True)
    tb(s, 0.9, 2.3, 11.5, 1.0, "Analysis reports", 44, WHITE, True)
    tb(s, 0.9, 3.4, 11.5, 1.0,
       f"The {len(rendered)} HTML reports of the ap_process_mining repository, reproduced in full",
       20, PALE)
    tb(s, 0.9, 5.6, 11.5, 0.9,
       ["Each report is shown as rendered in a browser; collapsed sections are opened.",
        "Source: github.com/ruckizuckiNO/ap_process_mining, commit 32b580a (28 Nov 2026)"],
       14, PALE)

    # index slide(s): numbers known once we know chunk counts
    first = 3  # title + index
    starts = []
    for sec, f, title, chunks in rendered:
        starts.append(first)
        first += len(chunks)
    s = prs.slides.add_slide(blank)
    tb(s, 0.6, 0.35, 12, 0.3, "CONTENTS", 11, TEAL, True)
    tb(s, 0.6, 0.62, 12, 0.6, "Reports in this appendix", 30, NAVY, True)
    rows = len(rendered) + 1
    t = s.shapes.add_table(rows, 4, Inches(0.6), Inches(1.4), Inches(12.15),
                           Inches(0.36 * rows)).table
    widths = (3.1, 4.6, 3.45, 1.0)
    for j, wd in enumerate(widths):
        t.columns[j].width = Inches(wd)
    head = ("Section", "Report", "File", "Slides")
    for j, h in enumerate(head):
        t.cell(0, j).text = h
    for i, ((sec, f, title, chunks), st) in enumerate(zip(rendered, starts), start=1):
        vals = (sec, title, Path(f).name, f"{st}–{st + len(chunks) - 1}")
        for j, v in enumerate(vals):
            t.cell(i, j).text = v
    for i in range(rows):
        t.rows[i].height = Inches(0.36)
        for j in range(4):
            c = t.cell(i, j)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (CARD if i % 2 else WHITE)
            for p_ in c.text_frame.paragraphs:
                for r in p_.runs:
                    r.font.name, r.font.size, r.font.bold = FONT, Pt(12), i == 0
                    r.font.color.rgb = WHITE if i == 0 else NAVY
    n = 2
    tb(s, 11.73, 7.12, 1.0, 0.25, str(n), 10, GREY, align=PP_ALIGN.RIGHT)

    # report slides
    for (sec, f, title, chunks) in rendered:
        for k, ch in enumerate(chunks, start=1):
            n += 1
            s = prs.slides.add_slide(blank)
            tb(s, 0.3, 0.16, 8.8, 0.35, title, 16, NAVY, True, anchor=MSO_ANCHOR.MIDDLE)
            tb(s, 9.1, 0.16, 3.93, 0.35, f"{sec.split(' ')[0]} · part {k} of {len(chunks)}",
               12, GREY, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
            sc = min(IMG_W / ch.width, IMG_H / ch.height)
            w, h = ch.width * sc, ch.height * sc
            buf = io.BytesIO()
            ch.save(buf, "JPEG", quality=85, optimize=True)
            buf.seek(0)
            pic = s.shapes.add_picture(buf, Inches(IMG_X + (IMG_W - w) / 2), Inches(IMG_Y),
                                       Inches(w), Inches(h))
            pic.name = f"{Path(f).stem} part {k}"
            tb(s, 0.3, 7.22, 10, 0.22, f, 10, TEAL, link=REPO + f)
            tb(s, 11.73, 7.22, 1.3, 0.22, str(n), 10, GREY, align=PP_ALIGN.RIGHT)

    prs.save(OUT)
    print("wrote", OUT, len(prs.slides), "slides", f"{OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
