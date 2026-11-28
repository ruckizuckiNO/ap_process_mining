"""Build output/appendix_reports.pptx: one appendix slide per HTML report.

Run: conda run -n datasci python notes/appendix_reports_pptx.py
Numbers are copied from the reports' short answers (commit 32b580a).
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "appendix_reports.pptx"
REPO = "https://github.com/ruckizuckiNO/ap_process_mining/blob/main/"

NAVY = RGBColor(0x14, 0x2B, 0x45)
TEAL = RGBColor(0x1F, 0x9E, 0xA3)
LIME = RGBColor(0xA3, 0xC9, 0x3A)
INK = RGBColor(0x1F, 0x29, 0x37)
GREY = RGBColor(0x5B, 0x67, 0x76)
CARD = RGBColor(0xE8, 0xF3, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
AMBER = RGBColor(0xB4, 0x5A, 0x09)
FONT = "Calibri"

REPORTS = [
    dict(
        sec="A.1 Definitions and clocks", file="docs/assumptions.html",
        title="Invoice assumptions",
        q="What counts as an invoice proxy, when the clock starts and stops, and how value and discount are set.",
        keys=[("2.9", "invoice proxies per ZNB1 PO on average; 53% of POs have more than one"),
              ("15.9M", "NOK missed discount, goods-receipt clock (headline)"),
              ("27.3M", "NOK with the arrival clock (upper bound)")],
        backs=["Invoice proxy: why a PO is not one invoice", "Measurement: elapsed time is not active work",
               "The start event changes the performance question"]),
    dict(
        sec="A.1 Definitions and clocks", file="output/report_assumptions_ctrl1.html",
        title="How much the CTRL1 assumptions matter",
        q="How sensitive is the CTRL1 missed discount to the clock start, the value split and the value field?",
        keys=[("11.97M", "NOK base case, 5,386 CTRL1 proxies posted manually"),
              ("+2.52M", "if the clock started 1 day earlier (14.49M)"),
              ("1 s", "median gap from arrival to validation")],
        backs=["Assumptions: measured, operationalised and proposed", "Every CTRL1 missed-discount figure"]),
    dict(
        sec="A.1 Definitions and clocks", file="output/report_invoice_clock.html",
        title="The invoice clock: ZJD1 vs ZNB1",
        q="Is ZNB1 really later than ZJD1 once both clocks start at goods receipt?",
        keys=[("13.1%", "of ZNB1 proxies post after 10 days, vs 6.4% for ZJD1"),
              ("15.9M", "NOK missed discount in ZNB1, vs 10.1M in ZJD1"),
              ("2.2%", "late in both types when a goods-receipt step exists")],
        backs=["We focused on ZNB1: 16.3M vs 11.9M", "The start event changes the performance question"]),
    dict(
        sec="A.1 Definitions and clocks", file="output/report_znb1_end_wait_gr.html",
        title="ZNB1: clock from END_WAIT_GR",
        q="Do proxies released from the goods-receipt wait look late only because of the clock?",
        keys=[("9.6%", "post after 10 days from END_WAIT_GR, vs 23.6% straight to CTRL1"),
              ("58.9%", "would look late counted from arrival"),
              ("8.0 days", "90th percentile, END_WAIT_GR to posting")],
        backs=["The start event changes the performance question", "WAITGR route (outside scope)"]),
    dict(
        sec="A.2 Process and routing", file="output/report.html",
        title="Invoice flow screening, all PO types",
        q="How do the 11 PO-type workflows behave, and where would an improvement project pay off?",
        keys=[("186,547", "invoice cases, 11 PO types, Jul–Nov 2025"),
              ("51% / 41%", "of PO value in ZJD1 / ZNB1")],
        img="notes/figures/po_value_by_type.png",
        backs=["ZNB1 was selected for control exposure", "Why ZNB1 rather than ZJD1?"]),
    dict(
        sec="A.2 Process and routing", file="output/report_znb1.html",
        title="ZNB1 invoice flow",
        q="What does the ZNB1 workflow look like, and how stable is it?",
        keys=[("13,776", "invoice proxies on 4,753 POs, 216 suppliers"),
              ("87%", "of proxies follow the three leading routes")],
        img="notes/figures/znb1_variants_pareto.png",
        backs=["Three leading routes account for 87%", "Whole-ZNB1 variant Pareto"]),
    dict(
        sec="A.2 Process and routing", file="output/report_ctrl1_direct.html",
        title="Proxies sent straight to the manual queue",
        q="How late are proxies routed to CTRL1 at validation, and what does it cost?",
        keys=[("22.9%", "post after 10 days, vs 5.6% for the rest of ZNB1"),
              ("12.7M", "NOK, 80% of ZNB1's missed discount")],
        img="notes/figures/znb1_ctrl1_lateness_histogram.png",
        backs=["We went on with ZNB1 and the manual queue", "CTRL1: 23% post after more than 10 days",
               "Most of the cost is lost between day 10 and 30"]),
    dict(
        sec="A.2 Process and routing", file="output/report_znb1_ctrl1.html",
        title="Why ZNB1 proxies go to the manual queue",
        q="What decides whether a ZNB1 proxy goes to CTRL1 or waits for goods receipt?",
        keys=[("None", "the log records no reason code for CTRL1"),
              ("59%", "of CTRL1 proxies come from 39 near-always-CTRL1 suppliers"),
              ("87%", "of later proxies follow their PO's first proxy")],
        backs=["How we identify relevant process problems", "Evidence synthesis: routing"],
        note="The 39 suppliers here (≥ 95% CTRL1) are not the 35 always-manual focus group (100%)."),
    dict(
        sec="A.2 Process and routing", file="output/report_znb1_ctrl1_model.html",
        title="How predictable is the manual queue?",
        q="Can a model predict the CTRL1 routing, and which fields carry the signal?",
        keys=[("0.95", "AUC with all fields, CV grouped by PO"),
              ("0.93", "AUC with the supplier alone; 0.81 without it"),
              ("0.69", "AUC at best for suppliers the model has not seen")],
        backs=["How we identify relevant process problems", "Evidence synthesis: routing"],
        note="The deck states an unseen-supplier AUC of 0.538; the report's best value is 0.69. Check which variant the deck means."),
    dict(
        sec="A.3 Queue and targets", file="output/report_queue_fifo.html",
        title="How the manual queue is worked",
        q="Is the manual queue worked first-in-first-out, and does anyone prioritise by value?",
        keys=[("0.94", "age rank for User1341: oldest proxy posted first"),
              ("145", "proxies waiting at a User1341 posting (median)"),
              ("8.63M", "of the focus group's 9.77M NOK missed discount")],
        backs=["Only one clerk works the queue FIFO", "The loss sits in the FIFO queue: 88%"]),
    dict(
        sec="A.3 Queue and targets", file="output/report_po_cut_gbr.html",
        title="PO-value cut: late-share model",
        q="If small POs ran like normal orders, how many would still be late?",
        keys=[("~5%", "still late below the cut (4.7–5.2%), vs 30% today"),
              ("0.88–1.12M", "NOK recovered at cuts of 1.5M to 2.25M")],
        img="notes/figures/znb1_ctrl1_100pct_po_cut.png",
        backs=["If small POs ran like normal orders", "The cut alone moves little money",
               "Threshold simulation"]),
    dict(
        sec="A.3 Queue and targets", file="output/report_scenario_po_cut.html",
        title="Target scenario: automate small POs, speed up the rest",
        q="How much missed discount is left if the three goals are met?",
        keys=[("0.76M", "NOK still missed, down from 9.97M (92% recovered)"),
              ("6.80M", "NOK of the recovery from the 69 POs above 5M")],
        img="output/figures/scenario_po_cut_waterfall.png",
        backs=["Our goals for the 35 always-manual suppliers", "The targets need a faster queue",
               "Which late proxies above 5M stay late"]),
    dict(
        sec="A.3 Queue and targets", file="output/report_control_charts.html",
        title="Control charts explained (optional)",
        q="Is the posting lead time in statistical control, week by week?",
        keys=[("207,081", "posted invoice proxies charted, 11 PO types"),
              ("11 of 11", "PO types out of control")],
        img="output/figures/control_charts/cc_ZNB1_by_clock.png",
        backs=["No slide yet: include only if the control charts go into the report"]),
]

GAPS = [
    ("Posting share by user, 35-supplier sample", "PNG only (notes/figures)"),
    ("Reconstructed backlog for User1341", "PNG only (notes/figures)"),
    ("Duration bands and cost by posting user", "PNG only (notes/figures)"),
    ("35 always-manual suppliers table (3,322 / 9.97M)", "no HTML has the 3,322 figure"),
    ("25-supplier threshold and exposure scenario", "older model, not the 35-supplier focus group"),
    ("report_zjd1_znb1_clock.html", "left out: same headline as report_invoice_clock.html"),
]


def text(slide, x, y, w, h, runs, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, margin=0):
    """Add a text box. runs: str or list of paragraphs (str or (str, dict))."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    paras = runs if isinstance(runs, list) else [runs]
    for i, p in enumerate(paras):
        t, opt = (p, {}) if isinstance(p, str) else p
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(opt.get("after", 0))
        r = para.add_run()
        r.text = t
        f = r.font
        f.name = FONT
        f.size = Pt(opt.get("size", size))
        f.bold = opt.get("bold", bold)
        f.color.rgb = opt.get("color", color)
        if opt.get("link"):
            r.hyperlink.address = opt["link"]
    return tb


def box(slide, x, y, w, h, fill, rounded=True):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    if rounded:
        shp.adjustments[0] = 0.08
    return shp


def footer(slide, n):
    text(slide, 0.6, 7.0, 8, 0.3, "INN354 · Group 10 · Appendix: supporting analysis reports",
         size=10, color=GREY)
    text(slide, 11.73, 7.0, 1.0, 0.3, str(n), size=10, color=GREY, align=PP_ALIGN.RIGHT)


def header(slide, kicker, title, file):
    text(slide, 0.6, 0.35, 12, 0.3, kicker.upper(), size=11, color=TEAL, bold=True)
    text(slide, 0.6, 0.62, 12.1, 0.6, title, size=30, color=NAVY, bold=True)
    text(slide, 0.6, 1.2, 12.1, 0.3, [(file, dict(link=REPO + file, color=TEAL))], size=12)


def backs_block(slide, x, y, w, backs):
    text(slide, x, y, w, 0.3, "Backs these exam slides", size=12, color=TEAL, bold=True)
    text(slide, x, y + 0.32, w, 0.3 * len(backs) + 0.1,
         [(f"→ {b}", dict(after=3)) for b in backs], size=12, color=INK)


def report_slide(prs, r, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, r["sec"], r["title"], r["file"])
    if r.get("img"):
        lx, lw = 0.6, 4.3
        text(s, lx, 1.7, lw, 0.3, "Question", size=12, color=TEAL, bold=True)
        text(s, lx, 1.98, lw, 0.9, r["q"], size=14)
        y = 3.0
        for big, lab in r["keys"]:
            text(s, lx, y, lw, 0.45, big, size=24, color=NAVY, bold=True)
            text(s, lx, y + 0.45, lw, 0.45, lab, size=12, color=GREY)
            y += 1.0
        backs_block(s, lx, y + 0.05, lw, r["backs"])
        # image, fitted into the right-hand area on a light card
        ax, ay, aw, ah = 5.25, 1.7, 7.5, 4.9
        box(s, ax, ay, aw, ah, CARD)
        pad = 0.15
        iw, ih = Image.open(ROOT / r["img"]).size
        sc = min((aw - 2 * pad) / iw, (ah - 2 * pad) / ih)
        w, h = iw * sc, ih * sc
        pic = s.shapes.add_picture(str(ROOT / r["img"]), Inches(ax + (aw - w) / 2),
                                   Inches(ay + (ah - h) / 2), Inches(w), Inches(h))
        pic.name = "chart"
        text(s, ax, ay + ah + 0.08, aw, 0.3, f"Chart: {r['img']}", size=10, color=GREY)
    else:
        cw, gap, cy, ch = 3.85, 0.35, 1.75, 2.0
        for i, (big, lab) in enumerate(r["keys"]):
            cx = 0.6 + i * (cw + gap)
            box(s, cx, cy, cw, ch, CARD)
            text(s, cx + 0.3, cy + 0.25, cw - 0.6, 0.75, big, size=40, color=NAVY, bold=True)
            text(s, cx + 0.3, cy + 1.05, cw - 0.6, 0.8, lab, size=14, color=INK)
        text(s, 0.6, 4.1, 5.9, 0.3, "Question", size=12, color=TEAL, bold=True)
        text(s, 0.6, 4.42, 5.9, 1.0, r["q"], size=16)
        backs_block(s, 6.85, 4.1, 5.9, r["backs"])
    if r.get("note"):
        nb = box(s, 0.6, 6.15, 12.15, 0.6, RGBColor(0xFD, 0xF3, 0xE4))
        text(s, 0.8, 6.15, 11.8, 0.6, [("Check: ", dict(bold=True, color=AMBER))], size=12,
             anchor=MSO_ANCHOR.MIDDLE)
        # second run on same line
        tf = s.shapes[-1].text_frame
        run = tf.paragraphs[0].add_run()
        run.text = r["note"]
        run.font.name, run.font.size, run.font.color.rgb = FONT, Pt(12), INK
    footer(s, n)


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1 title
    s = prs.slides.add_slide(blank)
    box(s, 0, 0, 13.333, 7.5, NAVY, rounded=False)
    text(s, 0.9, 1.0, 11, 0.4, "INN354 / SYSTEMS AND PROCESS INNOVATION · APPENDIX", size=14,
         color=LIME, bold=True)
    text(s, 0.9, 2.3, 11.5, 1.6, "Supporting analysis reports", size=44, color=WHITE, bold=True)
    text(s, 0.9, 3.45, 11.5, 1.0,
         "13 HTML reports from the ap_process_mining repository, mapped to the exam slides they back",
         size=20, color=RGBColor(0xC9, 0xDD, 0xE8))
    text(s, 0.9, 5.6, 11.5, 0.8,
         ["Every number on these slides is copied from the report it names.",
          ("Source: github.com/ruckizuckiNO/ap_process_mining, commit 32b580a (28 Nov 2026)",
           dict(link="https://github.com/ruckizuckiNO/ap_process_mining"))],
         size=14, color=RGBColor(0xC9, 0xDD, 0xE8))

    # 2 index
    s = prs.slides.add_slide(blank)
    text(s, 0.6, 0.35, 12, 0.3, "OVERVIEW", size=11, color=TEAL, bold=True)
    text(s, 0.6, 0.62, 12.1, 0.6, "Which report backs which part of the deck", size=30,
         color=NAVY, bold=True)
    rows = len(REPORTS) + 1
    tbl = s.shapes.add_table(rows, 3, Inches(0.6), Inches(1.4), Inches(12.15),
                             Inches(0.38 * rows)).table
    tbl.columns[0].width, tbl.columns[1].width, tbl.columns[2].width = (
        Inches(1.0), Inches(4.6), Inches(6.55))
    for j, h in enumerate(["Slide", "Report", "Backs"]):
        c = tbl.cell(0, j)
        c.text = h
    for i, r in enumerate(REPORTS, start=1):
        vals = [str(i + 2), r["file"].split("/")[-1], r["backs"][0]]
        for j, v in enumerate(vals):
            tbl.cell(i, j).text = v
    for i in range(rows):
        tbl.rows[i].height = Inches(0.38)
        for j in range(3):
            c = tbl.cell(i, j)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (CARD if i % 2 else WHITE)
            for p in c.text_frame.paragraphs:
                for run in p.runs:
                    run.font.name, run.font.size = FONT, Pt(12)
                    run.font.bold = i == 0
                    run.font.color.rgb = WHITE if i == 0 else INK
    footer(s, 2)

    for k, r in enumerate(REPORTS):
        report_slide(prs, r, k + 3)

    # last: gaps
    s = prs.slides.add_slide(blank)
    text(s, 0.6, 0.35, 12, 0.3, "GAPS", size=11, color=TEAL, bold=True)
    text(s, 0.6, 0.62, 12.1, 0.6, "Exam slides without an HTML report behind them", size=30,
         color=NAVY, bold=True)
    y = 1.55
    for what, why in GAPS:
        box(s, 0.6, y, 12.15, 0.68, CARD)
        text(s, 0.85, y, 6.6, 0.68, what, size=16, color=NAVY, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 7.5, y, 5.1, 0.68, why, size=14, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
        y += 0.8
    text(s, 0.6, 6.45, 12.15, 0.4,
         "The repository citation on the references slide names commit 344128b; the reports here are at 32b580a.",
         size=12, color=AMBER, bold=True)
    footer(s, len(REPORTS) + 3)

    prs.save(OUT)
    print("wrote", OUT, len(prs.slides), "slides")


if __name__ == "__main__":
    main()
