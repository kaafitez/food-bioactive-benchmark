#!/usr/bin/env python3
"""
make_review_figures.py — regenerate every figure in the bioactive-food-AI
review article from one shared style config.

USAGE
-----
    python make_review_figures.py         # regenerate all 4 figures into ./figures_review/
    python make_review_figures.py 3       # regenerate only Fig 3

Requires: matplotlib, pandas, numpy. Fig 3 also uses rdkit + networkx + adjustText
(pip install rdkit networkx adjustText); it degrades gracefully if adjustText is absent.

This is ONE self-contained script: all 4 review figures (three schematics —
graphical abstract, representation ladder, proposed architecture — plus one
data figure, the illustrative case study) are defined here, so there is nothing
else to import or keep in sync.

Only the data figure (Fig 3) reads external data: benchmark100_scores_det.csv
and ablation_scores_det.csv from DATA_DIR (default ./figdata). The three
schematics need no data files. Edit the STYLE / PALETTE and LAYOUT block below
to restyle every figure at once. The LAYOUT block is where you tune spacing,
padding and font sizes to taste:
    BASE_FS / TITLE_FS / PANEL_LAB_FS  — font sizes (pt)
    SUBPLOT_HS / SUBPLOT_WS            — gaps between panels within a figure
    LABEL_PAD / TITLE_PAD              — padding between axes and their labels/titles
    SAVE_PAD_IN                        — whitespace margin around the saved image
    LEGEND_PAD / BAR_WIDTH             — legend inset and bar thickness
    COL_1 / COL_2 / DPI                — figure widths (inches) and resolution
Changing any of these once at the top restyles all 4 figures on the next run.

DESIGN SYSTEM
-------------
* One colourblind-safe STRATUM palette threaded through EVERY figure that shows
  Western / East Asian / South Asian / African. African is reddish-purple, never
  alarm-red, so a data category is never confused with a warning.
* One MODEL palette (single-hue light->dark, because tiers are ordered).
* One COND palette for the four ablation grounding conditions.
* Sequential quantities (accuracy) use a perceptually-uniform sequential
  colormap (viridis-family), never a red-green diverging map.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def _title(obj, *a, **k):
    """Gate a title/subtitle behind SHOW_TITLES."""
    return obj.set_title(*a, **k) if SHOW_TITLES else None
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D
# Fig 3 (chemical-similarity network) needs rdkit + networkx; the other figures do not.
try:
    import networkx as nx
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _HAVE_CHEM = True
except ImportError:
    _HAVE_CHEM = False
try:
    from adjustText import adjust_text
    _HAVE_ADJUSTTEXT = True
except ImportError:
    _HAVE_ADJUSTTEXT = False

# ----------------------------------------------------------------------------
# STYLE / PALETTE  — edit here to restyle all figures
# ----------------------------------------------------------------------------
DPI          = 300
FONT         = "Times New Roman"  # restyle: serif, per author spec
BASE_FS      = 8.5                # base font size (pt)
TITLE_FS     = 9.5
PANEL_LAB_FS = 11
PANEL_CASE   = "upper"   # "upper" -> A,B,C (user preference); "lower" -> a,b,c (Nature Portfolio house style)                 # a/b/c panel letters
SHOW_TITLES  = False              # no in-figure titles/subtitles (captions carry them)
COL_1        = 3.5                # single-column width (inches)
COL_2        = 7.2                # double-column width (inches)

# --- LAYOUT: spacing, padding, margins (tune to taste) ---------------------
# These feed matplotlib's constrained/tight layout and per-figure spacing.
PAD          = 0.4                # global padding around each figure (pts, w_pad/h_pad)
H_PAD        = 0.4                # extra height padding between stacked panels
W_PAD        = 0.4                # extra width padding between side-by-side panels
SUBPLOT_HS   = 0.35              # hspace: vertical gap between subplots (fraction of axis height)
SUBPLOT_WS   = 0.30              # wspace: horizontal gap between subplots
SAVE_PAD_IN  = 0.06              # savefig bbox_inches padding (inches) around the tight box
LABEL_PAD    = 3.0                # axis label padding (pts)
TITLE_PAD    = 6.0                # axis title padding (pts)
LEGEND_PAD   = 0.4                # legend borderaxespad
BAR_WIDTH    = 0.72              # default bar width where bars are drawn

# Cultural strata — colourblind-safe (Okabe-Ito derived), NO alarm-red
STRATUM = {
    "Western":     "#4878A8",   # dusty blue      (muted, pleasing)
    "East Asian":  "#5B9279",   # muted teal-green
    "South Asian": "#E0A458",   # warm sand
    "African":     "#B5533D",   # muted brick (CVD-safe vs Western blue; was dusty rose #B0729A)
}
STRATUM_ORDER = ["Western", "East Asian", "South Asian", "African"]
# short aliases used in some data files
STRATUM_ALIAS = {"East": "East Asian", "South": "South Asian",
                 "EastAsian": "East Asian", "SouthAsian": "South Asian"}

# Model tiers — single-hue sequential (ordered by capability)
MODEL = {
    "Haiku-4.5":  "#c6dbef",
    "Sonnet-4.5": "#6baed6",
    "Opus-4.5":   "#2171b5",
    "Sonnet-5":   "#08306b",
}
MODEL_ORDER = ["Haiku-4.5", "Sonnet-4.5", "Opus-4.5", "Sonnet-5"]

# Ablation grounding conditions
COND = {"A": "#9a9a9a", "B": "#4878A8", "C": "#E0A458", "D": "#5B9279"}
COND_LABEL = {"A": "baseline", "B": "+structure", "C": "+provenance", "D": "both"}

# Sequential colormap for accuracy heatmaps (perceptually uniform, CVD-safe)
ACC_CMAP = "viridis"

# ----------------------------------------------------------------------------
# DATA  — version ids resolved to local paths at runtime
# ----------------------------------------------------------------------------
# Data-directory search path: works from the figure bundle (figdata/) AND from a
# clone of the released repository (data/), so a reviewer can regenerate every
# figure from a clean checkout with no edits. Override with FIGDATA=<dir>.
DATA_DIRS = [os.environ["FIGDATA"]] if os.environ.get("FIGDATA") else \
            ["figdata", "data", "../data", ".", "code/figdata"]
# Each key lists accepted filenames in priority order (bundle name, then the
# clean names used in the public repository).
DATA = {
    "crossfam":        ["crossfamily_summary.csv"],
    "bench100":        ["benchmark100_scores_det.csv", "benchmark_scores.csv"],
    "ablation":        ["ablation_scores_det.csv", "ablation_scores.csv"],
    "compounds":       ["compounds_200.csv", "compounds.csv"],
    "compounds100":    ["bioactive_benchmark_set.csv", "compounds.csv"],
    "crossfam_stats":  ["crossfamily_stats.json"],
    "comparison":      ["compound_set_comparison.json"],
    "taskdecomp":      ["ablation_task_decomposition.json"],
    "gapclose":        ["gap_closure_decomposition.json"],
    "hardened":        ["hardened_stats.json"],
    "structure_probe": ["structure_probe_results.csv"],
    "accuracy_ba":     ["model_accuracy_beforeafter.json"],
}
OUT_DIR = os.environ.get("OUTDIR", "figures_review")


def D(key):
    """Load a data file (csv->DataFrame, json->dict).

    Searches DATA_DIRS for any of the accepted filenames for `key`, so the
    script runs unchanged from the figure bundle (figdata/) or a clone of the
    released repository (data/).
    """
    candidates = DATA[key] if isinstance(DATA[key], (list, tuple)) else [DATA[key]]
    for d in DATA_DIRS:
        for fname in candidates:
            path = os.path.join(d, fname)
            if os.path.exists(path):
                if path.endswith(".csv"):
                    return pd.read_csv(path)
                with open(path) as f:
                    return json.load(f)
    raise FileNotFoundError(
        f"data for '{key}' not found; looked for {candidates} in {DATA_DIRS}")


def norm_stratum(s):
    return STRATUM_ALIAS.get(s, s)


def setup():
    mpl.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI,
        "font.family": FONT, "font.size": BASE_FS,
        "axes.titlesize": TITLE_FS, "axes.labelsize": BASE_FS,
        "xtick.labelsize": BASE_FS - 0.5, "ytick.labelsize": BASE_FS - 0.5,
        "legend.fontsize": BASE_FS - 0.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "savefig.bbox": "tight", "savefig.pad_inches": SAVE_PAD_IN,
        "savefig.facecolor": "white",
        "axes.labelpad": LABEL_PAD, "axes.titlepad": TITLE_PAD,
        "figure.subplot.hspace": SUBPLOT_HS, "figure.subplot.wspace": SUBPLOT_WS,
        "legend.borderaxespad": LEGEND_PAD,
        "figure.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel_label(ax, letter, dx=-0.02, dy=1.04):
    letter = letter.upper() if PANEL_CASE == "upper" else letter.lower()
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=PANEL_LAB_FS,
            fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p)
    plt.close(fig)
    print("  saved", p)
    return p

def _light(hexc, f=0.80):
    r = int(hexc[1:3], 16); g = int(hexc[3:5], 16); b = int(hexc[5:7], 16)
    r = int(r + (255 - r) * f); g = int(g + (255 - g) * f); b = int(b + (255 - b) * f)
    return f"#{r:02x}{g:02x}{b:02x}"

NAVY = "#1a1a1a"   # dark text accent (titles gated off)


def fig1():
    """Graphical abstract framed as the review's argument: field -> three gaps -> path forward."""
    fig, ax = plt.subplots(figsize=(COL_2 * 1.5, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off")
    if SHOW_TITLES:
        ax.text(6, 6.05, "Food AI can taste, but it cannot yet reason about health",
                fontsize=13.5, fontweight="bold", ha="center", color=NAVY)
        ax.text(6, 5.62, "A review of language models for food, the gaps that remain, and a path toward bioactive-aware systems",
                fontsize=9, ha="center", color="#555", style="italic")

    def box(x, y, w, h, txt, fc, ec, fs=8, bold=False, tc="#222"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                     fc=fc, ec=ec, lw=1.4, zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal", zorder=3)

    # COLUMN 1 — what food AI already does (the reviewed landscape)
    ax.text(2.05, 4.98, "WHERE FOOD AI WORKS TODAY", fontsize=8.3, fontweight="bold",
            color="#777", ha="center")
    for k, (yy, lab) in enumerate([(4.35,"Flavour & sensory association"),
                                    (3.75,"Sensor / vision safety & quality"),
                                    (3.15,"Nutrition & recipe text"),
                                    (2.55,"Generative food & recipe design")]):
        box(0.3, yy, 3.5, 0.5, lab, "#e8f0f7", STRATUM["Western"], 7.5)
    box(0.3, 1.75, 3.5, 0.62, "Bioactive compounds that\ndetermine health: largely absent",
        _light(STRATUM["African"], 0.78), STRATUM["African"], 7.8, bold=True, tc="#2b2b2b")

    ax.add_patch(FancyArrowPatch((3.95, 3.2), (4.6, 3.2), arrowstyle="-|>",
                 mutation_scale=18, color="#333", lw=2))

    # COLUMN 2 — the three organizing gaps (the review's synthesis)
    ax.text(6.5, 4.98, "THREE ORGANIZING GAPS", fontsize=8.3, fontweight="bold",
            color="#777", ha="center")
    box(4.75, 4.2, 3.5, 0.66, "1  Bioactive chemistry\nis absent", "#fbeef4", STRATUM["African"], 7.8, bold=True, tc="#2b2b2b")
    box(4.75, 3.42, 3.5, 0.66, "2  Structure treated\nas text, not a graph", "#eaf1fb", STRATUM["Western"], 7.8, bold=True, tc="#2b2b2b")
    box(4.75, 2.64, 3.5, 0.66, "3  Culture measured for\ncuisines, not compounds", "#fff4e6", STRATUM["South Asian"], 7.8, bold=True, tc="#b9770e")
    ax.text(6.5, 2.28, "gaps 2 and 3 are downstream of gap 1", fontsize=6.6,
            ha="center", color="#888", style="italic")

    ax.add_patch(FancyArrowPatch((8.4, 3.2), (9.05, 3.2), arrowstyle="-|>",
                 mutation_scale=18, color="#333", lw=2))

    # COLUMN 3 — the path forward (the review's prescription)
    ax.text(10.35, 4.98, "A PATH FORWARD", fontsize=8.3, fontweight="bold",
            color="#777", ha="center")
    box(9.2, 4.05, 2.55, 0.8, "Provenance as a\nfirst-class input\n(retrieval-grounded)",
        _light(STRATUM["East Asian"], 0.72), STRATUM["East Asian"], 7.4, bold=True, tc="#2b2b2b")
    box(9.2, 3.05, 2.55, 0.8, "Molecular-graph &\nspectral encoders\n(next hypothesis)",
        "#eeeeee", "#999", 7.2, tc="#666")
    box(9.2, 2.05, 2.55, 0.8, "Bioactives as the\nobject of analysis",
        "#eaf1fb", STRATUM["Western"], 7.4, bold=True, tc="#2b2b2b")

    # bottom strip — the one piece of illustrative evidence, clearly labelled as such
    ax.plot([0.35, 11.75], [1.5, 1.5], color="#ddd", lw=0.8)
    ax.text(6, 1.15, "Illustrative evidence (case study): across four LLMs, accuracy on health-relevant "
            "tasks falls ~23 points from Western to African",
            fontsize=7.6, ha="center", color="#555")
    ax.text(6, 0.75, "bioactives; supplying cultural provenance closes most of that gap, while added model scale does not.",
            fontsize=7.6, ha="center", color="#555")
    return save(fig, "fig01_graphical_abstract.png")

def fig2():
    fig, ax = plt.subplots(figsize=(COL_2, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
    if SHOW_TITLES:
        ax.text(5, 5.95, "The representation ladder for food bioactives", fontsize=TITLE_FS + 1,
                fontweight="bold", ha="center", color=NAVY)
    rungs = [
        ("Compound name", "“curcumin”", "text token; what every food LLM uses", "#f2f2f2", "#999"),
        ("SMILES string", "COc1cc(...)", "text token sequence; structure implicit", "#e8f0f7", STRATUM["Western"]),
        ("2D molecular graph", "atoms + bonds", "explicit connectivity (GNN encoder)", _light(STRATUM["East Asian"], 0.7), "#1e7d46"),
        ("3D + provenance", "conformer + origin", "structure and cultural context together", _light(STRATUM["East Asian"], 0.5), "#145a32"),
    ]
    y = 0.6
    for i, (title, mid, sub, fc, ec) in enumerate(rungs):
        ax.add_patch(FancyBboxPatch((1.2 + i * 0.25, y), 6.5, 1.0, boxstyle="round,pad=0.03",
                     fc=fc, ec=ec, lw=1.5, zorder=2 + i))
        ax.text(1.5 + i * 0.25, y + 0.5, f"{title}", fontsize=BASE_FS + 0.5, fontweight="bold",
                va="center", color="#222", zorder=10)
        ax.text(7.5 + i * 0.25, y + 0.72, mid, fontsize=BASE_FS - 1.5, va="center", ha="right",
                color="#555", family="monospace", zorder=10)
        ax.text(7.5 + i * 0.25, y + 0.28, sub, fontsize=BASE_FS - 2, va="center", ha="right",
                color="#777", style="italic", zorder=10)
        y += 1.2
    ax.annotate("", xy=(0.85, 5.0), xytext=(0.85, 0.7),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.8))
    ax.text(0.55, 2.9, "increasing chemical fidelity", rotation=90, va="center", ha="center",
            fontsize=BASE_FS - 1, color="#444")
    return save(fig, "fig02_representation_ladder.png")


def fig3():
    """Structure does not sort food bioactives by culture (two-panel, no crowding).

    a: cultural composition of each broad chemical class (every class is a mix of origins).
    b: pairwise Tanimoto similarity for same-culture vs different-culture compound pairs
       (the two distributions nearly coincide, so structure barely encodes culture).
    Replaces the earlier 100-node labelled network, which was inherently crowded; the
    labelled class-grouped network is retained as a standalone talk asset
    (fig3_similarity_network_byclass.png). Requires rdkit; scipy for the KDE."""
    if not _HAVE_CHEM:
        print("  [fig3] skipped: rdkit/networkx not installed")
        return None
    from scipy.stats import gaussian_kde
    THRESH = 0.30
    df = D("compounds100")
    df["stratum"] = df["stratum"].map(norm_stratum)

    def family(cls):
        c = str(cls).lower()
        if any(k in c for k in ["flavon", "flavan", "anthocyan", "isoflav", "chalcone",
                                 "catechin", "theaflavin", "biflavon"]): return "Flavonoids"
        if "curcumin" in c or "diarylheptanoid" in c: return "Curcuminoids"
        if "stilben" in c: return "Stilbenoids"
        if any(k in c for k in ["hydroxycinnamic", "phenolic acid", "phenylpropanoid",
                                 "phenylethanoid", "polyphenolic", "ellagitannin", "ellagi"]):
            return "Phenolic acids / tannins"
        if "lignan" in c: return "Lignans"
        if any(k in c for k in ["triterpen", "saponin", "limonoid", "cycloartane", "dammarane"]):
            return "Triterpenoids / saponins"
        if any(k in c for k in ["diterpen", "abietane", "labdane", "kaurane"]): return "Diterpenoids"
        if any(k in c for k in ["carotenoid", "monoterpen", "sesquiterpen"]): return "Other terpenoids"
        if any(k in c for k in ["alkaloid", "purine", "piperidine", "isoquinoline", "sceletium"]):
            return "Alkaloids"
        if any(k in c for k in ["organosulfur", "isothiocyanate"]): return "Organosulfur"
        return "Other"

    df["family"] = df["compound_class"].map(family)
    strata_order = ["Western", "East Asian", "South Asian", "African"]
    M = pd.crosstab(df["family"], df["stratum"]).reindex(columns=strata_order, fill_value=0)
    fam_order = list(M.sum(axis=1).sort_values(ascending=True).index)

    # pairwise Tanimoto: same-culture vs cross-culture
    smis = df["canonical_smiles"].fillna(df["smiles"]).tolist()
    mols = [Chem.MolFromSmiles(s) for s in smis]
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(m) if m is not None else None for m in mols]
    strat = df["stratum"].tolist()
    same_c, cross_c = [], []
    for a in range(len(fps)):
        if fps[a] is None:
            continue
        for b in range(a + 1, len(fps)):
            if fps[b] is None:
                continue
            t = DataStructs.TanimotoSimilarity(fps[a], fps[b])
            (same_c if strat[a] == strat[b] else cross_c).append(t)
    same_c = np.array(same_c); cross_c = np.array(cross_c)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(COL_2, COL_2 * 0.49),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})
    # panel a: stacked cultural composition
    y = np.arange(len(fam_order)); left = np.zeros(len(fam_order))
    for s in strata_order:
        vals = M.reindex(fam_order)[s].values
        axA.barh(y, vals, left=left, color=STRATUM[s], edgecolor="white",
                 linewidth=0.6, label=s, height=0.74)
        left = left + vals
    axA.set_yticks(y); axA.set_yticklabels(fam_order, fontsize=BASE_FS - 2.3)
    axA.set_xlabel("Number of compounds", fontsize=BASE_FS - 1)
    axA.set_xlim(0, max(35, int(M.sum(axis=1).max()) + 1))
    axA.legend(fontsize=BASE_FS - 2.7, loc="lower right", frameon=False,
               title="Cultural origin", title_fontsize=BASE_FS - 2.3,
               handlelength=1.0, labelspacing=0.3)
    for sp in ["top", "right"]:
        axA.spines[sp].set_visible(False)
    axA.text(-0.02, 1.04, "A" if PANEL_CASE=="upper" else "a", transform=axA.transAxes, fontsize=PANEL_LAB_FS,
             fontweight="bold", va="bottom", ha="right")

    # panel b: same vs cross culture similarity KDE
    xs = np.linspace(0, 0.8, 200)
    ks, kc = gaussian_kde(same_c), gaussian_kde(cross_c)
    axB.fill_between(xs, kc(xs), color="#c4c4c4", alpha=0.9, lw=0,
                     label="Different culture (n=%d)" % len(cross_c))
    axB.plot(xs, ks(xs), color="#1a1a1a", lw=1.7,
             label="Same culture (n=%d)" % len(same_c))
    axB.axvline(cross_c.mean(), color="#888", ls=(0, (4, 2)), lw=0.9)
    axB.axvline(same_c.mean(), color="#1a1a1a", ls=(0, (4, 2)), lw=0.9)
    ymax = float(kc(xs).max())
    axB.annotate("mean %.2f" % same_c.mean(),
                 xy=(float(same_c.mean()), float(np.atleast_1d(ks(same_c.mean()))[0])),
                 xytext=(0.37, ymax * 0.82), fontsize=BASE_FS - 2.5, color="#1a1a1a",
                 arrowprops=dict(arrowstyle="-", color="#1a1a1a", lw=0.6))
    axB.annotate("mean %.2f" % cross_c.mean(),
                 xy=(float(cross_c.mean()), float(np.atleast_1d(kc(cross_c.mean()))[0]) * 0.55),
                 xytext=(0.30, ymax * 0.36), fontsize=BASE_FS - 2.5, color="#666",
                 arrowprops=dict(arrowstyle="-", color="#888", lw=0.6))
    axB.set_xlabel("Pairwise Tanimoto similarity", fontsize=BASE_FS - 1)
    axB.set_ylabel("Density", fontsize=BASE_FS - 1)
    axB.set_xlim(0, 0.8)
    axB.legend(fontsize=BASE_FS - 2.7, loc="upper right", frameon=False, labelspacing=0.3)
    for sp in ["top", "right"]:
        axB.spines[sp].set_visible(False)
    axB.text(-0.02, 1.04, "B" if PANEL_CASE=="upper" else "b", transform=axB.transAxes, fontsize=PANEL_LAB_FS,
             fontweight="bold", va="bottom", ha="right")

    fig.tight_layout(w_pad=1.6)
    return save(fig, "fig03_similarity_network.png")



def fig4():
    """Illustrative case study: cultural gap, where it concentrates, and its closure."""
    import numpy as np
    from matplotlib.gridspec import GridSpec
    b = D("bench100"); ab = D("ablation")
    b["stratum"] = b["stratum"].map(norm_stratum)
    ab["stratum"] = ab["stratum"].map(norm_stratum)
    order = STRATUM_ORDER

    fig = plt.figure(figsize=(7.4, 3.0))
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.30, 0.82], wspace=0.72, figure=fig)

    # a: accuracy falls with cultural distance
    axa = fig.add_subplot(gs[0])
    pooled = b.groupby("stratum")["overall"].mean().reindex(order)
    for mdl in sorted(b["model"].unique()):
        md = b[b.model == mdl].groupby("stratum")["overall"].mean().reindex(order)
        axa.plot(range(4), md.values, color="#c7c7c7", lw=0.8, marker="o", ms=2.3, zorder=1)
    axa.plot(range(4), pooled.values, color="#08306b", lw=2.2, marker="o", ms=5.5, zorder=3,
             label="mean of 4 tiers")
    gap = pooled["Western"] - pooled["African"]
    axa.annotate("", xy=(3, pooled["African"]), xytext=(3, pooled["Western"]),
                 arrowprops=dict(arrowstyle="<->", color="#555", lw=1.0))
    axa.text(2.8, (pooled["Western"] + pooled["African"]) / 2, f"{gap*100:.0f}-pt\ngap",
             fontsize=BASE_FS - 2.2, va="center", ha="right", color="#555")
    axa.set_xticks(range(4)); axa.set_xticklabels(["West", "E.Asia", "S.Asia", "Africa"], rotation=20, ha="right")
    axa.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9]); axa.set_ylim(0.45, 0.97)
    axa.set_ylabel("overall accuracy")
    _title(axa, "Accuracy falls with\ncultural distance", fontsize=BASE_FS)
    axa.legend(frameon=False, fontsize=BASE_FS - 2.5, loc="upper right", bbox_to_anchor=(1.02, 1.02))
    axa.text(0.03, 0.04, "\u2191 higher = better", transform=axa.transAxes,
             fontsize=BASE_FS - 2.5, color="#888")
    panel_label(axa, "a")

    # b: stratum x task heatmap (where the gap concentrates)
    axb = fig.add_subplot(gs[1])
    tasks = {"correct_bioactivity": "bioactivity", "correct_food": "food\nsource",
             "correct_cultural": "cultural\ncontext", "correct_bioavail": "bio-\navail."}
    M = b.groupby("stratum")[list(tasks)].mean().reindex(order)
    im = axb.imshow(M.values, cmap="cividis", vmin=0.2, vmax=1.0, aspect="auto")
    axb.set_xticks(range(4)); axb.set_xticklabels(list(tasks.values()), fontsize=BASE_FS - 2.3)
    axb.set_yticks(range(4)); axb.set_yticklabels(order, fontsize=BASE_FS - 1.7)
    for i in range(4):
        for j in range(4):
            v = M.values[i, j]
            axb.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=BASE_FS - 2.1,
                     color="white" if v < 0.62 else "black", fontweight="bold")
    _title(axb, "Where the gap concentrates:\nAfrican cultural context is the floor", fontsize=BASE_FS)
    cb = fig.colorbar(im, ax=axb, fraction=0.05, pad=0.04, ticks=[0.2, 0.6, 1.0])
    cb.set_label("accuracy", fontsize=BASE_FS - 2.2); cb.ax.tick_params(labelsize=BASE_FS - 2.7)
    axb.set_xticks(np.arange(-0.5, 4, 1), minor=True); axb.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    axb.grid(which="minor", color="white", lw=1.2); axb.tick_params(which="minor", length=0)
    panel_label(axb, "b")

    # c: provenance closes the gap
    axc = fig.add_subplot(gs[2])
    gaps = {}
    for c in ["A", "C"]:
        sub = ab[ab.condition == c]
        gaps[c] = (sub[sub.stratum == "Western"]["overall"].mean()
                   - sub[sub.stratum == "African"]["overall"].mean())
    axc.bar([0, 1], [gaps["A"], gaps["C"]], color=[COND["A"], COND["C"]], width=0.6,
            edgecolor="white", lw=0.8)
    axc.axhline(0, color="black", lw=0.7)
    axc.set_xticks([0, 1]); axc.set_xticklabels(["no\ncontext", "+ prove-\nnance"], fontsize=BASE_FS - 2)
    axc.set_ylabel("Western \u2212 African gap"); axc.set_ylim(0, 0.30)
    _title(axc, "Provenance\ncloses the gap", fontsize=BASE_FS)
    for i, c in enumerate(["A", "C"]):
        axc.text(i, gaps[c] + 0.008, f"{gaps[c]:.2f}", ha="center", fontsize=BASE_FS - 1.7, fontweight="bold")
    red = (gaps["A"] - gaps["C"]) / gaps["A"] * 100
    axc.text(0.5, 0.265, f"\u2212{red:.0f}%", ha="center", fontsize=BASE_FS - 1.5,
             color="#2b2b2b", fontweight="bold")
    panel_label(axc, "c")

    return save(fig, "fig04_illustrative_benchmark.png")

def fig5():
    """Before/after accuracy for all six model families: reveals the floor effect that
    the gap-only view hides (a small gap can mean a weak model OR a strong one)."""
    D_ = D("accuracy_ba")
    order = sorted(D_, key=lambda m: -D_[m]["A"]["Overall"])        # strongest -> weakest
    disp = {"Mixtral-8x7B": "Mixtral-8\u00d77B"}
    W_COL, AF_COL = STRATUM["Western"], STRATUM["African"]           # shared muted palette
    GAP_COL, MID = "#c2c2c2", "#8a8a8a"
    INK = "#1a1a1a"                                                  # dark text
    n = len(order)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(COL_2, 4.4), sharey=True)
    for _pi, (cond, ptitle, ax) in enumerate([("A", "Baseline\n(name + structure only)", axL),
                             ("C", "+ Provenance\n(retrieved origin note)", axR)]):
        panel_label(ax, "ab"[_pi], dx=-0.04, dy=1.02)
        for k, m in enumerate(order):
            y = n - 1 - k
            w, af = D_[m][cond]["Western"], D_[m][cond]["African"]
            ea, sa = D_[m][cond]["East Asian"], D_[m][cond]["South Asian"]
            ax.plot([af, w], [y, y], color=GAP_COL, lw=2.6, zorder=1, solid_capstyle="round")
            for v in (ea, sa):
                ax.plot([v], [y], marker="|", ms=8, mec=MID, mew=1.4, zorder=2)
            ax.plot([w], [y], "o", ms=8.5, color=W_COL, zorder=3)
            ax.plot([af], [y], "o", ms=8.5, color=AF_COL, zorder=3)
            gap = w - af
            ax.text((af + w) / 2, y + 0.30, f"{gap:.2f}" if gap >= 0 else f"\u2212{abs(gap):.2f}",
                    ha="center", va="bottom", fontsize=BASE_FS - 2.4, color=INK)
        _title(ax, ptitle, fontsize=BASE_FS, pad=6)
        ax.set_xlim(0, 0.9); ax.set_xticks([0, .25, .5, .75])
        ax.set_xticklabels(["0", "25%", "50%", "75%"], fontsize=BASE_FS - 2)
        ax.set_ylim(-0.6, n - 0.4); ax.set_xlabel("Accuracy", fontsize=BASE_FS - 1)
        ax.grid(axis="x", lw=0.5, alpha=0.25); ax.set_axisbelow(True)
        for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
        ax.tick_params(left=False)

    axL.set_yticks(range(n))
    axL.set_yticklabels([f"{disp.get(m, m)}\n{D_[m]['A']['Overall']*100:.0f}% overall"
                         for m in order[::-1]], fontsize=BASE_FS - 2)
    # honest annotations: small gap can mean weak (floor) or strong
    axL.annotate("smallest gap, but lowest\naccuracy \u2014 a floor effect,\nnot fairness",
                 xy=(D_["Mixtral-8x7B"]["A"]["African"], 0), xytext=(0.015, 0.95),
                 fontsize=BASE_FS - 2.6, color=INK, va="center",
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    axL.annotate("small gap because\nstrong everywhere",
                 xy=(D_["Claude"]["A"]["African"], n - 1), xytext=(0.055, n - 1.15),
                 fontsize=BASE_FS - 2.6, color=INK, va="center",
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    axR.text(0.89, -0.5, "higher = better", fontsize=BASE_FS - 2.6, style="italic",
             color="#555", ha="right")

    leg = [Line2D([0], [0], marker="o", color="w", mfc=W_COL, ms=8, label="Western accuracy"),
           Line2D([0], [0], marker="o", color="w", mfc=AF_COL, ms=8, label="African accuracy"),
           Line2D([0], [0], marker="|", color=MID, ms=9, mew=1.4, label="East / South Asian"),
           Line2D([0], [0], color=GAP_COL, lw=2.6, label="Western\u2013African gap")]
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 0.90 if SHOW_TITLES else 0.98),
               ncol=4, frameon=False, fontsize=BASE_FS - 2.2, handletextpad=0.4, columnspacing=1.5)
    if SHOW_TITLES:
        fig.suptitle("Accuracy before and after adding cultural provenance", fontsize=BASE_FS + 1.5, y=1.0)
    fig.subplots_adjust(top=0.80 if SHOW_TITLES else 0.88, wspace=0.08, left=0.15, right=0.98, bottom=0.11)
    return save(fig, "fig05_crossfamily_heatmap.png")



def fig6():
    fig, ax = plt.subplots(figsize=(COL_2 * 1.5, 6.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

    def box(x, y, w, h, txt, fc, ec, fs=8.5, tc="black", bold=False, lw=1.3, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                     fc=fc, ec=ec, lw=lw, zorder=2, linestyle=ls))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal", zorder=3)

    def arrow(x1, y1, x2, y2, c="#333", lw=1.6, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=14, color=c, lw=lw, zorder=1, linestyle=ls))

    ax.text(1.5, 6.75, "INPUTS", ha="center", fontsize=9, fontweight="bold", color="#555")
    # provenance (validated) drawn FIRST/top with solid emphasis; structure/spectral dashed (untested)
    box(0.2, 5.35, 2.6, 0.95, "Cultural-provenance\nmetadata (compound-level\norigin, indigenous use)",
        _light(STRATUM["East Asian"], 0.78), STRATUM["East Asian"], 8, bold=True)
    box(0.2, 4.05, 2.6, 0.95, "Molecular structure\n(2D/3D graph)", "#f2f2f2", "#aaa", 8, ls="--", tc="#666")
    box(0.2, 2.75, 2.6, 0.95, "Analytical chemistry\nLC-MS / NMR / FTIR", "#f2f2f2", "#aaa", 8, ls="--", tc="#666")
    box(0.2, 1.45, 2.6, 0.95, "Food-matrix &\nprocessing context", "#f2f2f2", "#aaa", 8, ls="--", tc="#666")

    ax.text(4.6, 6.75, "MODALITY ENCODERS", ha="center", fontsize=9, fontweight="bold", color="#555")
    box(3.7, 5.35, 1.9, 0.95, "Provenance\nembedding", _light(STRATUM["East Asian"], 0.68),
        "#1e7d46", 8.5, bold=True)
    box(3.7, 4.05, 1.9, 0.95, "Graph neural\nnetwork encoder", "#ededed", "#999", 8.3, ls="--", tc="#666")
    box(3.7, 2.75, 1.9, 0.95, "Spectral / sequence\nencoder", "#ededed", "#999", 8.3, ls="--", tc="#666")
    box(3.7, 1.45, 1.9, 0.95, "Structured-context\nencoder", "#ededed", "#999", 8.3, ls="--", tc="#666")

    ax.text(7.35, 6.75, "BRIDGE / PROJECTION", ha="center", fontsize=9, fontweight="bold", color="#555")
    box(6.55, 2.55, 1.6, 3.75, "Cross-modal\nbridge module\n\n(Q-Former /\nprojection into\nLLM token space)",
        "#ece3f6", "#7a4fb0", 8.3, bold=True)

    ax.text(10.2, 6.75, "REASONING BACKBONE", ha="center", fontsize=9, fontweight="bold", color="#555")
    box(9.1, 3.4, 2.4, 2.9, "LLM backbone\n(frozen or\nLoRA-tuned)\n\nGrounded reasoning\nover fused tokens",
        "#e9e9e9", "#555", 8.6, bold=True)
    box(9.1, 1.15, 2.7, 1.7, "OUTPUTS\n• bioactivity class\n• food source & matrix\n• cultural context of use\n• bioavailability / metabolism",
        _light(STRATUM["East Asian"], 0.85), "#3f7a34", 8)

    for y in [5.82, 4.52, 3.22, 1.92]:
        solid = (y > 5.5)
        arrow(2.8, y, 3.7, y, c="#333" if solid else "#aaa", ls="-" if solid else "--")
    for y in [5.82, 4.52, 3.22, 1.92]:
        solid = (y > 5.5)
        arrow(5.6, y, 6.55, 4.4 if y > 4 else 3.6, c="#333" if solid else "#aaa", ls="-" if solid else "--")
    arrow(8.15, 4.4, 9.1, 4.85)
    arrow(10.3, 3.4, 10.4, 2.85)
    ax.add_patch(FancyArrowPatch((9.1, 2.0), (6.9, 2.4), connectionstyle="arc3,rad=-0.3",
                 arrowstyle="-|>", mutation_scale=11, color="#999", lw=1.1, ls="--", zorder=1))
    ax.text(7.7, 1.55, "provenance-grounded\nverification", fontsize=6.8, color="#888",
            ha="center", style="italic")
    # legend for solid vs dashed
    ax.text(0.2, 0.5, "Solid = empirically validated (provenance).   Dashed = proposed but untested at this data scale "
            "(graph / spectral / matrix encoders).", fontsize=7.2, color="#444", style="italic")
    return save(fig, "fig06_architecture.png")


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6}

if __name__ == "__main__":
    setup()
    import sys
    want = [int(a) for a in sys.argv[1:]] or sorted(FIGS)
    for n in want:
        print(f"Fig {n}:")
        FIGS[n]()
