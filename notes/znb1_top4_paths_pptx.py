"""The four most frequent ZNB1 process paths (variants per invoice) as a PowerPoint file.

Writes notes/znb1_top4_process_paths.pptx with plain OOXML (standard library only; python-pptx/pptxgenjs are not
available in this environment). Numbers come from the pipeline outputs (clock: ZNB1 from goods receipt where WAIT_GR).
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_top4_paths_pptx.py
"""
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

from common import SLA_SOFT_DAYS, load_events, load_invoices

NOTES = Path(__file__).resolve().parent
OUT = NOTES / "znb1_top4_process_paths.pptx"
K = ["PO_ID", "SAP_PO_No"]
BOTS = {"User1587", "User280"}

# --- Data -------------------------------------------------------------------------------------
ev = load_events()
z = ev[ev.SAP_PO_Type_Code == "ZNB1"].sort_values(K + ["ts", "Event_ID"]).copy()
inv = load_invoices().set_index(K)
z["gap_days"] = (z.groupby(K).ts.shift(-1) - z.ts).dt.total_seconds() / 86400
variants = z.groupby(K).Activity_ID.agg(tuple)
n_all = len(variants)
top = variants.value_counts().head(4)
paths = []
for v, n in top.items():
    ids = variants[variants == v].index
    s = z.set_index(K).loc[ids]
    gaps = s.groupby("Activity_ID", sort=False).gap_days.median()
    who = s.groupby("Activity_ID", sort=False).User_ID.agg(lambda u: "bot" if u.isin(BOTS).mean() > 0.5 else "person")
    d = inv.loc[ids].time_to_post_days
    arrival = (inv.loc[ids].first_post - inv.loc[ids].start).dt.total_seconds() / 86400
    route = s[s.Activity_ID == "Initial validation"].SAP_Next_Processing_Type.value_counts()
    paths.append({"steps": list(v), "n": int(n), "share": n / n_all, "gaps": [gaps.get(a) for a in v[:-1]],
                  "who": [who[a] for a in v], "posted": int(d.notna().sum()), "median_days": d.median(),
                  "median_arrival": arrival.median(), "late": (d > SLA_SOFT_DAYS).mean() if d.notna().any() else None,
                  "route": route})
cover = top.sum() / n_all

# --- OOXML helpers --------------------------------------------------------------------------------
EMU = 914400
W, H = 13.333, 7.5
NAVY, INK, BODY, MUTED = "14213D", "1F1F1E", "4A5568", "6B6A64"
BOT_FILL, WAIT, LINE, WHITE, GREY = "D6E4F5", "B4501A", "CFCBC0", "FFFFFF", "E4E3DC"
NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')


def e(x):
    return int(round(x * EMU))


class Slide:
    def __init__(self, bg=WHITE):
        self.bg, self.items, self.nid = bg, [], 2

    def _id(self):
        self.nid += 1
        return self.nid

    @staticmethod
    def _runs(paras, align):
        out = []
        for runs in paras:
            rr = []
            for text, size, bold, color in runs:
                rr.append(f'<a:r><a:rPr lang="en-US" sz="{int(size * 100)}" b="{1 if bold else 0}" dirty="0">'
                          f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
                          f'<a:latin typeface="Arial"/><a:cs typeface="Arial"/></a:rPr><a:t>{escape(text)}</a:t></a:r>')
            out.append(f'<a:p><a:pPr algn="{align}"/>{"".join(rr)}</a:p>')
        return "".join(out)

    def shape(self, x, y, w, h, paras=None, fill=None, line=None, geom="rect", align="l", anchor="t", txbox=False,
              name="Shape", inset=0.08, line_w=1.25):
        i = self._id()
        fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else "<a:noFill/>"
        line_xml = (f'<a:ln w="{int(line_w * 12700)}"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
                    if line else "<a:ln><a:noFill/></a:ln>")
        av = '<a:avLst><a:gd name="adj" fmla="val 18000"/></a:avLst>' if geom == "roundRect" else "<a:avLst/>"
        body = ""
        if paras:
            ins = e(inset)
            body = (f'<p:txBody><a:bodyPr wrap="square" lIns="{ins}" tIns="{ins}" rIns="{ins}" bIns="{ins}" '
                    f'anchor="{anchor}"><a:normAutofit/></a:bodyPr><a:lstStyle/>{self._runs(paras, align)}</p:txBody>')
        else:
            body = '<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang="en-US" dirty="0"/></a:p></p:txBody>'
        tb = ' txBox="1"' if txbox else ""
        self.items.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="{escape(name)} {i}"/><p:cNvSpPr{tb}/>'
            f'<p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{e(x)}" y="{e(y)}"/><a:ext cx="{e(w)}" cy="{e(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="{geom}">{av}</a:prstGeom>{fill_xml}{line_xml}</p:spPr>{body}</p:sp>')

    def text(self, x, y, w, h, paras, align="l", anchor="t", name="Text"):
        self.shape(x, y, w, h, paras, align=align, anchor=anchor, txbox=True, name=name, inset=0)

    def arrow(self, x1, y, x2, color=MUTED, name="Arrow"):
        i = self._id()
        self.items.append(
            f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{i}" name="{name} {i}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{e(x1)}" y="{e(y)}"/><a:ext cx="{e(x2 - x1)}" cy="0"/></a:xfrm>'
            f'<a:prstGeom prst="straightConnector1"><a:avLst/></a:prstGeom>'
            f'<a:ln w="22225"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:tailEnd type="triangle"/></a:ln>'
            f'</p:spPr></p:cxnSp>')

    def xml(self):
        return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sld {NS}><p:cSld><p:bg><p:bgPr>'
                f'<a:solidFill><a:srgbClr val="{self.bg}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree>'
                '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm>'
                '<a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm>'
                f'</p:grpSpPr>{"".join(self.items)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/>'
                '</p:clrMapOvr></p:sld>')


LABEL = {"Data management": "Data management", "Initial validation": "Initial validation",
         "Wait for goods receipt": "Wait for goods receipt", "Posted manually": "Clerk posts manually",
         "MM Posted Automatically": "Bot posts automatically", "Invoice document deleted": "Deleted"}
LETTERS = "ABCD"


def step_box(sl, x, y, w, h, step, who, size, end=False):
    if end:
        sl.shape(x, y, w, h, [[(step, size, True, BODY)]], fill=GREY, geom="roundRect", align="ctr", anchor="ctr",
                 name="End")
    elif who == "bot":
        sl.shape(x, y, w, h, [[(LABEL.get(step, step), size, True, NAVY)]], fill=BOT_FILL, geom="roundRect",
                 align="ctr", anchor="ctr", name="Bot step")
    else:
        sl.shape(x, y, w, h, [[(LABEL.get(step, step), size, True, NAVY)]], fill=WHITE, line=NAVY, line_w=1.75,
                 geom="roundRect", align="ctr", anchor="ctr", name="Person step")


def flow(sl, p, x0, y, bw, bh, gap, size, lab_size):
    """Steps left to right; median days between steps above each arrow; a grey end box if never posted."""
    steps, who, gaps = list(p["steps"]), list(p["who"]), list(p["gaps"])
    end = p["posted"] == 0
    if end:
        steps.append("Stopped – never posted")
        who.append("end")
        gaps.append(None)
    x = x0
    for k, (s_, w_) in enumerate(zip(steps, who)):
        step_box(sl, x, y, bw, bh, s_, w_, size, end=(w_ == "end"))
        if k < len(steps) - 1:
            g = gaps[k]
            lab = "" if g is None or pd.isna(g) else f"{g:.1f} d" if g >= 0.05 else "0 d"
            color = WAIT if (g is not None and not pd.isna(g) and g >= 1) else MUTED
            sl.arrow(x + bw + 0.06, y + bh / 2, x + bw + gap - 0.06, color=color)
            if lab:
                sl.text(x + bw, y + bh / 2 - 0.36, gap, 0.3, [[(lab, lab_size, g >= 1, color)]], align="ctr")
        x += bw + gap


def pct(v):
    return f"{100 * v:.1f}%"


slides = []

# 1 Title
s = Slide(NAVY)
s.text(0.8, 1.7, 11.5, 0.4, [[("INN354 · AP PROCESS MINING · ZNB1 MANUAL ORDERS", 14, True, "F0A877")]])
s.text(0.8, 2.25, 11.5, 1.6, [[("The four most frequent ZNB1 process paths", 40, True, WHITE)]])
s.text(0.8, 3.95, 11.5, 1.0, [[(f"{n_all:,} invoices, July 2025 – March 2026. The four paths cover "
                                f"{100 * cover:.1f}% of them.", 20, False, "C9D1E0")]])
s.text(0.8, 6.4, 11.5, 0.4, [[("Source: ZNB1 event log, one variant per invoice (SAP_PO_No). Medians in days.", 12,
                               False, "AEB6C8")]])
slides.append(s)

# 2 Overview
s = Slide()
s.text(0.6, 0.45, 12.1, 0.7, [[(f"Four paths cover {100 * cover:.0f}% of ZNB1 invoices", 32, True, NAVY)]])
s.text(0.6, 1.12, 12.1, 0.4, [[("Steps per invoice; numbers on arrows = median days between steps", 14, False,
                                MUTED)]])
row_y, row_h = 1.75, 1.0
for k, p in enumerate(paths):
    y = row_y + k * (row_h + 0.18)
    s.shape(0.6, y + 0.2, 0.6, 0.6, [[(LETTERS[k], 20, True, WHITE)]], fill=NAVY, geom="ellipse", align="ctr",
            anchor="ctr", name="Badge", inset=0)
    s.text(1.35, y + 0.12, 1.3, 0.8, [[(pct(p["share"]), 20, True, NAVY)], [(f"{p['n']:,} invoices", 11, False,
                                                                              MUTED)]])
    flow(s, p, 2.7, y + 0.18, 1.7, 0.64, 0.55, 11, 10)
    if p["posted"]:
        right = [[(f"{p['median_days']:.1f} d", 18, True, NAVY)],
                 [(f"median · {100 * p['late']:.0f}% late", 11, False, MUTED)]]
    else:
        right = [[("never posted", 14, True, BODY)], [("not in lead times", 11, False, MUTED)]]
    s.text(11.45, y + 0.12, 1.5, 0.8, right)
ly = 6.65
s.shape(0.6, ly, 0.3, 0.22, fill=BOT_FILL, geom="roundRect", name="Legend bot")
s.text(0.98, ly - 0.02, 1.4, 0.3, [[("Bot step", 11, False, BODY)]])
s.shape(2.35, ly, 0.3, 0.22, fill=WHITE, line=NAVY, geom="roundRect", name="Legend person")
s.text(2.73, ly - 0.02, 1.6, 0.3, [[("Person step", 11, False, BODY)]])
s.text(4.4, ly - 0.02, 8.4, 0.5, [[("Orange = waiting ≥ 1 day. Days to posting counted from goods receipt where "
                                    "booked, else from arrival; late = more than 10 days.", 11, False, MUTED)]])
slides.append(s)

# 3–6 One slide per path
TITLES = ["straight to the manual queue", "goods receipt matches, bot posts",
          "waits for goods receipt, clerk posts", "stopped at validation"]
TEXT = [
    ["Initial validation sends the invoice straight to the manual queue (CTRL1), skipping the goods-receipt check.",
     "It waits about a week until a clerk posts it; posting itself takes about a minute.",
     "Most of it is User1341’s supplier portfolio, worked in a weekly, oldest-first rhythm."],
    ["The goods receipt is already booked when the invoice arrives.",
     "The bot matches invoice, PO and goods receipt and posts within seconds – no person involved.",
     "Counted from goods receipt, none of these invoices is late."],
    ["The invoice arrives before the goods receipt and waits about three weeks for it.",
     "When the receipt is booked it does not match (price, quantity), so a clerk posts it – usually within hours.",
     "The delay sits with purchasing and the warehouse, not with AP."],
    ["Validation stops the invoice; it is never posted.",
     "Most are routed to deletion (duplicates, wrong PO reference or similar); a few were still open when the log ends.",
     "No payment term applies, so these invoices are left out of all lead-time figures."],
]
for k, p in enumerate(paths):
    s = Slide()
    s.text(0.6, 0.45, 12.1, 0.7, [[(f"Path {LETTERS[k]}: {TITLES[k]}", 30, True, NAVY)]])
    s.text(0.6, 1.1, 12.1, 0.4, [[(f"{p['n']:,} invoices · {pct(p['share'])} of ZNB1", 14, False, MUTED)]])
    nsteps = len(p["steps"]) + (1 if p["posted"] == 0 else 0)
    bw, gap = 2.2, 0.85
    total = nsteps * bw + (nsteps - 1) * gap
    flow(s, p, (W - total) / 2, 1.95, bw, 0.9, gap, 14, 13)
    cards = [(f"{p['n']:,}", f"invoices ({pct(p['share'])})")]
    if p["posted"]:
        if k == 2:
            cards.append((f"{p['median_arrival']:.0f} d", f"median from arrival · {p['median_days']:.1f} d from "
                                                         "goods receipt"))
        else:
            cards.append((f"{p['median_days']:.1f} d", "median days to posting"))
        cards.append((f"{100 * p['late']:.0f}%", "posted after more than 10 days"))
    else:
        dele = p["route"].get("DELE", 0)
        cards.append((f"{100 * dele / p['n']:.0f}%", f"routed to deletion ({dele:,})"))
        cards.append(("0", "posted – excluded from lead times"))
    cw, cg = 3.75, 0.4
    cx0 = (W - (3 * cw + 2 * cg)) / 2
    for j, (big, lab) in enumerate(cards):
        s.shape(cx0 + j * (cw + cg), 3.45, cw, 1.45, [[(big, 30, True, NAVY)], [(lab, 13, False, BODY)]],
                fill="F2F1EC", geom="roundRect", anchor="ctr", name="Stat card", inset=0.22)
    s.text(cx0, 5.2, 3 * cw + 2 * cg, 1.3, [[("What happens: ", 15, True, NAVY), (" ".join(TEXT[k]), 15, False,
                                                                                    BODY)]])
    s.text(0.6, 6.85, 12.1, 0.3, [[("Numbers on arrows = median days between steps. Clock: from goods receipt where "
                                    "booked, else from arrival in AP. Source: ZNB1 event log.", 10, False, MUTED)]])
    slides.append(s)

# --- Package --------------------------------------------------------------------------------------
THEME = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="ZNB1"><a:themeElements>
<a:clrScheme name="ZNB1"><a:dk1><a:srgbClr val="{INK}"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
<a:dk2><a:srgbClr val="{NAVY}"/></a:dk2><a:lt2><a:srgbClr val="F2F1EC"/></a:lt2>
<a:accent1><a:srgbClr val="2A78D6"/></a:accent1><a:accent2><a:srgbClr val="{WAIT}"/></a:accent2>
<a:accent3><a:srgbClr val="1BAF7A"/></a:accent3><a:accent4><a:srgbClr val="EDA100"/></a:accent4>
<a:accent5><a:srgbClr val="{BOT_FILL}"/></a:accent5><a:accent6><a:srgbClr val="{MUTED}"/></a:accent6>
<a:hlink><a:srgbClr val="2A78D6"/></a:hlink><a:folHlink><a:srgbClr val="{NAVY}"/></a:folHlink></a:clrScheme>
<a:fontScheme name="Arial"><a:majorFont><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>
<a:minorFont><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>
<a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
<a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>
<a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>
<a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>
<a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>
<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle>
<a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>'''

EMPTY_TREE = ('<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
              '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/>'
              '<a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>')
MASTER = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster {NS}>{EMPTY_TREE}'
          '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" '
          'accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
          '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst></p:sldMaster>')
LAYOUT_TREE = EMPTY_TREE.replace("<p:cSld>", '<p:cSld name="Blank">')
LAYOUT = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout {NS} type="blank" preserve="1">'
          f'{LAYOUT_TREE}<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def rels(items):
    body = "".join(f'<Relationship Id="{i}" Type="{REL}/{t}" Target="{tg}"/>' for i, t, tg in items)
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships '
            f'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{body}</Relationships>')


n = len(slides)
pres = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentation {NS} saveSubsetFonts="1">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>'
        + "".join(f'<p:sldId id="{256 + i}" r:id="rId{i + 3}"/>' for i in range(n))
        + f'</p:sldIdLst><p:sldSz cx="{e(W)}" cy="{e(H)}"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>')
ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types '
      'xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
      '<Default Extension="xml" ContentType="application/xml"/>'
      '<Override PartName="/ppt/presentation.xml" '
      'ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
      '<Override PartName="/ppt/slideMasters/slideMaster1.xml" '
      'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
      '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" '
      'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
      '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
      '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
      '<Override PartName="/docProps/app.xml" '
      'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
      + "".join(f'<Override PartName="/ppt/slides/slide{i + 1}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
                for i in range(n)) + "</Types>")
core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>ZNB1 process paths</dc:title>'
        '</cp:coreProperties>')
app = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties '
       'xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
       f'<Application>Microsoft Office PowerPoint</Application><Slides>{n}</Slides></Properties>')

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("[Content_Types].xml", ct)
    zf.writestr("_rels/.rels", rels([("rId1", "officeDocument", "ppt/presentation.xml")]).replace(
        "</Relationships>",
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/'
        'core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" '
        f'Type="{REL}/extended-properties" Target="docProps/app.xml"/></Relationships>'))
    zf.writestr("docProps/core.xml", core)
    zf.writestr("docProps/app.xml", app)
    zf.writestr("ppt/presentation.xml", pres)
    zf.writestr("ppt/_rels/presentation.xml.rels", rels(
        [("rId1", "slideMaster", "slideMasters/slideMaster1.xml"), ("rId2", "theme", "theme/theme1.xml")]
        + [(f"rId{i + 3}", "slide", f"slides/slide{i + 1}.xml") for i in range(n)]))
    zf.writestr("ppt/slideMasters/slideMaster1.xml", MASTER)
    zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", rels(
        [("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml"), ("rId2", "theme", "../theme/theme1.xml")]))
    zf.writestr("ppt/slideLayouts/slideLayout1.xml", LAYOUT)
    zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels",
                rels([("rId1", "slideMaster", "../slideMasters/slideMaster1.xml")]))
    zf.writestr("ppt/theme/theme1.xml", THEME)
    for i, sl in enumerate(slides):
        zf.writestr(f"ppt/slides/slide{i + 1}.xml", sl.xml())
        zf.writestr(f"ppt/slides/_rels/slide{i + 1}.xml.rels",
                    rels([("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml")]))

print(f"wrote {OUT} ({n} slides); top-4 cover {100 * cover:.1f}% of {n_all:,} invoices")
for k, p in enumerate(paths):
    print(LETTERS[k], p["n"], f"{100 * p['share']:.1f}%", " > ".join(p["steps"]), p["gaps"], p["median_days"],
          p["late"])
