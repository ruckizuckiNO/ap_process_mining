"""Short appendix deck: only the report sections behind the exam PDF (28 Nov 2026).

Writes output/appendix_html_reports_short.pptx. Reuses the rendering and cutting
of appendix_html_pptx.py. Section choices are by index of the report's content-
column children (header = 0); rerun notes after a report's layout changes.
Run: conda run -n datasci python -W ignore notes/appendix_html_short_pptx.py
"""
import io
from pathlib import Path

from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

import appendix_html_pptx as base
from appendix_html_pptx import (CARD, FONT, GREY, IMG_H, IMG_W, IMG_X, IMG_Y, LIME, NAVY, PALE,
                                REPO, ROOT, SCALE, TEAL, WHITE, rect, tb)

OUT = ROOT / "output" / "appendix_html_reports_short.pptx"

# (section, file, kept child ranges, what is kept, exam slides it backs)
REPORTS = [
    ("A.1", "docs/assumptions.html", [(0, 0), (2, 9)],
     "Definitions 1–7, effect on the ZNB1 headline",
     "Invoice proxy; Measurement; Assumptions; The start event changes the question"),
    ("A.1", "output/report_assumptions_ctrl1.html", [(0, 2), (8, 20), (25, None)],
     "Short answer, clock, value split, summary",
     "Assumptions (2% discount, clock); CTRL1 missed-discount figures"),
    ("A.1", "output/report_invoice_clock.html", [(0, 2), (13, 32)],
     "Short answer, ZJD1 vs ZNB1, is the GR clock fair?",
     "Why ZNB1 (16.3M vs 11.9M, 13.1% vs 6.4%); The start event"),
    ("A.2", "output/report_znb1.html", [(0, 0), (2, 5), (17, 25), (29, 32)],
     "PO value per type, top variants, cost by variant, routing, who works on ZNB1",
     "ZNB1 selected for control exposure; Three leading routes; The potential sits in CTRL1; "
     "Recorded posting activity"),
    ("A.2", "output/report_ctrl1_direct.html", [(0, 16), (17, 19)],
     "Short answer, clock, ZJD1 vs ZNB1, who posts, suppliers",
     "We went on with CTRL1 (12.7M); CTRL1: 23% after 10 days; Cost between day 10 and 30; "
     "Duration bands by user"),
    ("A.2", "output/report_znb1_ctrl1.html", [(0, 2), (8, 15)],
     "Short answer, the supplier decides, the decision sticks to the PO",
     "How we identify process problems (routing); Evidence synthesis"),
    ("A.2", "output/report_znb1_ctrl1_model.html", [(0, 2), (8, 15), (22, 25)],
     "Short answer, results, robustness (unseen suppliers)",
     "Routing prediction AUC 0.931 / 0.954; Evidence synthesis"),
    ("A.3", "output/report_queue_fifo.html", [(0, None)],
     "Whole report",
     "Only one clerk works FIFO; The loss sits in the FIFO queue (88%)"),
    ("A.3", "output/report_po_cut_gbr.html", [(0, 4), (17, 26)],
     "Short answer, setup, results per cut, sensitivity",
     "If small POs ran like normal orders; The cut alone moves little money; Threshold simulation"),
    ("A.3", "output/report_scenario_po_cut.html", [(0, 19)],
     "Short answer, scenario, result, gap to targets, sensitivity",
     "Our goals; The targets need a faster queue; Which late proxies above 5M stay late"),
]
SECNAME = {"A.1": "A.1 Definitions and clocks", "A.2": "A.2 Process and routing",
           "A.3": "A.3 Queue and targets"}


def main():
    rendered = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=dict(width=1280, height=900), device_scale_factor=SCALE)
        for sec, f, keep, kept, backs in REPORTS:
            title, img, h2 = base.render(page, ROOT / f, keep)
            chunks = base.cut(base.trim_bottom(img), h2)
            rendered.append((sec, f, title, kept, backs, chunks))
            print(f"{f}: {len(chunks)} slides")
        browser.close()

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    s = prs.slides.add_slide(blank)
    rect(s, 0, 0, 13.333, 7.5, NAVY)
    tb(s, 0.9, 1.0, 11, 0.4, "INN354 / SYSTEMS AND PROCESS INNOVATION · APPENDIX", 14, LIME, True)
    tb(s, 0.9, 2.3, 11.5, 1.0, "Analysis reports behind the slides", 44, WHITE, True)
    tb(s, 0.9, 3.4, 11.5, 1.0,
       f"The sections of {len(rendered)} HTML reports that back figures in the main presentation",
       20, PALE)
    tb(s, 0.9, 5.6, 11.5, 0.9,
       ["Shown as rendered in a browser; sections not used in the presentation are left out.",
        "Full reports: github.com/ruckizuckiNO/ap_process_mining, commit 32b580a (28 Nov 2026)"],
       14, PALE)

    starts, nxt = [], 3
    for r in rendered:
        starts.append(nxt)
        nxt += len(r[-1])
    s = prs.slides.add_slide(blank)
    tb(s, 0.6, 0.3, 12, 0.3, "CONTENTS", 11, TEAL, True)
    tb(s, 0.6, 0.55, 12, 0.6, "What each report backs in the presentation", 28, NAVY, True)
    rows = len(rendered) + 1
    t = s.shapes.add_table(rows, 4, Inches(0.4), Inches(1.3), Inches(12.53),
                           Inches(0.52 * rows)).table
    for j, wd in enumerate((3.2, 3.6, 4.83, 0.9)):
        t.columns[j].width = Inches(wd)
    for j, h in enumerate(("Report", "Sections kept", "Backs these slides", "Slides")):
        t.cell(0, j).text = h
    for i, ((sec, f, title, kept, backs, chunks), st) in enumerate(zip(rendered, starts), 1):
        end = st + len(chunks) - 1
        for j, v in enumerate((title, kept, backs, f"{st}–{end}" if end > st else str(st))):
            t.cell(i, j).text = v
    for i in range(rows):
        t.rows[i].height = Inches(0.36 if i == 0 else 0.52)
        for j in range(4):
            c = t.cell(i, j)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (CARD if i % 2 else WHITE)
            for p_ in c.text_frame.paragraphs:
                for r in p_.runs:
                    r.font.name, r.font.size, r.font.bold = FONT, Pt(10.5), i == 0 or j == 0
                    r.font.color.rgb = WHITE if i == 0 else NAVY
    tb(s, 11.73, 7.22, 1.3, 0.22, "2", 10, GREY, align=PP_ALIGN.RIGHT)

    n = 2
    for sec, f, title, kept, backs, chunks in rendered:
        for k, ch in enumerate(chunks, start=1):
            n += 1
            s = prs.slides.add_slide(blank)
            tb(s, 0.3, 0.16, 8.8, 0.35, title, 16, NAVY, True, anchor=MSO_ANCHOR.MIDDLE)
            tb(s, 9.1, 0.16, 3.93, 0.35, f"{sec} · part {k} of {len(chunks)}", 12, GREY,
               align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
            sc = min(IMG_W / ch.width, IMG_H / ch.height)
            w, h = ch.width * sc, ch.height * sc
            buf = io.BytesIO()
            ch.save(buf, "JPEG", quality=85, optimize=True)
            buf.seek(0)
            pic = s.shapes.add_picture(buf, Inches(IMG_X + (IMG_W - w) / 2), Inches(IMG_Y),
                                       Inches(w), Inches(h))
            pic.name = f"{Path(f).stem} part {k}"
            tb(s, 0.3, 7.22, 10, 0.22, f"{f} (excerpt)", 10, TEAL, link=REPO + f)
            tb(s, 11.73, 7.22, 1.3, 0.22, str(n), 10, GREY, align=PP_ALIGN.RIGHT)

    prs.save(OUT)
    print("wrote", OUT, len(prs.slides), "slides", f"{OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
