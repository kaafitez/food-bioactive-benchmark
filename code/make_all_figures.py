#!/usr/bin/env python3
"""
make_all_figures.py — regenerate every figure in the bioactive-food-AI review
from one shared style config.

USAGE
-----
    python make_all_figures.py            # regenerate all figures into ./figures/
    python make_all_figures.py 6 10       # regenerate only Fig 6 and Fig 10

This is ONE self-contained script: all 12 figures (the 9 data figures plus the
3 schematic diagrams — graphical abstract, representation ladder, architecture)
are defined here, so there is nothing else to import or keep in sync.

All figures read their data from the CSV/JSON files listed in DATA (paths are
resolved relative to DATA_DIR, default ./figdata). Edit the STYLE / PALETTE and
LAYOUT block below to restyle every figure at once. The LAYOUT block is where you
tune spacing, padding and font sizes to taste:
    BASE_FS / TITLE_FS / PANEL_LAB_FS  — font sizes (pt)
    SUBPLOT_HS / SUBPLOT_WS            — gaps between panels within a figure
    LABEL_PAD / TITLE_PAD              — padding between axes and their labels/titles
    SAVE_PAD_IN                        — whitespace margin around the saved image
    LEGEND_PAD / BAR_WIDTH             — legend inset and bar thickness
    COL_1 / COL_2 / DPI                — figure widths (inches) and resolution
Changing any of these once at the top restyles all 12 figures on the next run.

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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path

# ----------------------------------------------------------------------------
# STYLE / PALETTE  — edit here to restyle all figures
# ----------------------------------------------------------------------------
DPI          = 300
FONT         = "DejaVu Sans"      # swap to 'Arial'/'Helvetica' if installed
BASE_FS      = 8.5                # base font size (pt)
TITLE_FS     = 9.5
PANEL_LAB_FS = 11                 # a/b/c panel letters
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
    "Western":     "#0072B2",   # blue
    "East Asian":  "#009E73",   # bluish-green
    "South Asian": "#E69F00",   # orange
    "African":     "#CC79A7",   # reddish-purple
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
COND = {"A": "#999999", "B": "#56B4E9", "C": "#E69F00", "D": "#009E73"}
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
    "bench100":        ["benchmark100_scores_det.csv", "benchmark_scores.csv"],
    "ablation":        ["ablation_scores_det.csv", "ablation_scores.csv"],
    "compounds":       ["compounds_200.csv", "compounds.csv"],
    "compounds100":    ["bioactive_benchmark_set.csv", "compounds.csv"],
    "crossfam":        ["crossfamily_summary.csv"],
    "crossfam_stats":  ["crossfamily_stats.json"],
    "comparison":      ["compound_set_comparison.json"],
    "taskdecomp":      ["ablation_task_decomposition.json"],
    "gapclose":        ["gap_closure_decomposition.json"],
    "hardened":        ["hardened_stats.json"],
    "structure_probe": ["structure_probe_results.csv"],
}
OUT_DIR = "figures"


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
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=PANEL_LAB_FS,
            fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p)
    plt.close(fig)
    print("  saved", p)
    return p


# ============================================================================
# FIG 3 — benchmark compound set composition (200-set)  [4 panels]
# ============================================================================
def fig3():
    """Composition: alluvial flow (stratum->activity->bioavailability) + chemical space."""
    df = D("compounds")
    df["stratum"] = df["stratum"].map(norm_stratum)
    fig = plt.figure(figsize=(COL_2, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], hspace=0.38, wspace=0.28)

    # top: alluvial spanning both columns
    ax_al = fig.add_subplot(gs[0, :])
    _draw_alluvial(ax_al, df)
    ax_al.set_title("a   Dataset composition flows across stratum, activity and bioavailability",
                    fontsize=TITLE_FS, loc="left", pad=10)

    # bottom-left: chemical space (MW vs logP), coloured by stratum, Lipinski box
    axb = fig.add_subplot(gs[1, 0])
    for s_ in STRATUM_ORDER:
        sub = df[df.stratum == s_]
        axb.scatter(sub.mw, sub.logp, s=10, color=STRATUM[s_], alpha=0.75,
                    linewidths=0, label=s_)
    axb.axvline(500, ls=":", color="grey", lw=0.8)
    axb.axhline(5, ls=":", color="grey", lw=0.8)
    axb.set_xlabel("molecular weight (Da)")
    axb.set_ylabel("cLogP")
    axb.set_title("b   Chemical space", loc="left")
    axb.legend(fontsize=BASE_FS - 2.5, loc="upper right", handletextpad=0.2,
               labelspacing=0.25, markerscale=1.3)

    # bottom-right: bioavailability by stratum, sequential greys (ordered, no red-green)
    axc = fig.add_subplot(gs[1, 1])
    tiers = ["low", "moderate", "high"]
    tier_col = {"low": "#d9d9d9", "moderate": "#969696", "high": "#252525"}
    bottom = np.zeros(4)
    for t in tiers:
        vals = [((df.stratum == s_) & (df.bioavailability == t)).sum() for s_ in STRATUM_ORDER]
        axc.bar(range(4), vals, bottom=bottom, color=tier_col[t], label=t,
                edgecolor="white", linewidth=0.5, width=0.7)
        bottom += vals
    axc.set_xticks(range(4))
    axc.set_xticklabels([s_.split()[0] for s_ in STRATUM_ORDER], rotation=30, ha="right")
    axc.set_ylabel("compounds")
    axc.set_title("c   Bioavailability tiers", loc="left")
    axc.legend(fontsize=BASE_FS - 2, loc="lower center", bbox_to_anchor=(0.5, -0.42),
               ncol=3, handlelength=0.9, columnspacing=1.0, title=None)

    return save(fig, "fig03_dataset.png")


# ============================================================================
# FIG 5 — bioactive-property accuracy by task and model (grouped bars)
# ============================================================================
def fig5():
    df = D("bench100")
    tasks = ["correct_bioactivity", "correct_food", "correct_cultural", "correct_bioavail"]
    tlab = ["bioactivity", "food\nsource", "cultural\ncontext", "bioavail."]
    fig, ax = plt.subplots(figsize=(COL_2 * 0.62, 2.6))
    w = 0.2
    x = np.arange(len(tasks))
    for i, mdl in enumerate(MODEL_ORDER):
        sub = df[df.model == mdl]
        means = [sub[t].mean() for t in tasks]
        ax.bar(x + (i - 1.5) * w, means, w, color=MODEL[mdl], label=mdl,
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(tlab)
    ax.set_ylabel("accuracy (↑ better)")
    ax.set_ylim(0, 1)
    ax.axhline(df[tasks].mean().mean(), ls="--", color="grey", lw=0.8, zorder=0)
    ax.set_title("Accuracy by task and model tier (100 compounds)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=BASE_FS-1.5, columnspacing=1.0, handlelength=1.0)
    fig.tight_layout()
    return save(fig, "fig05_task_model.png")


# ============================================================================
# FIG 6 — the cultural accuracy gap  [a: slope by tier | b: sequential heatmap]
# ============================================================================
def fig6():
    df = D("bench100")
    df["stratum"] = df["stratum"].map(norm_stratum)
    fig, axs = plt.subplots(1, 2, figsize=(COL_2, 2.7),
                            gridspec_kw={"width_ratios": [1, 1.15]})

    # a: accuracy vs stratum, one line per model tier (MODEL palette)
    for mdl in MODEL_ORDER:
        sub = df[df.model == mdl]
        ys = [sub[sub.stratum == s]["overall"].mean() for s in STRATUM_ORDER]
        axs[0].plot(range(4), ys, "o-", color=MODEL[mdl], label=mdl, lw=1.6, ms=4)
    axs[0].set_xticks(range(4))
    axs[0].set_xticklabels([s.split()[0] for s in STRATUM_ORDER], rotation=30, ha="right")
    axs[0].set_ylabel("overall accuracy (↑ better)")
    axs[0].set_ylim(0.4, 1.0)
    axs[0].set_title("Accuracy falls with cultural distance")
    axs[0].legend(fontsize=BASE_FS - 2, loc="lower left")
    # gap annotation
    afr = [df[(df.model == m) & (df.stratum == "African")]["overall"].mean() for m in MODEL_ORDER]
    wes = [df[(df.model == m) & (df.stratum == "Western")]["overall"].mean() for m in MODEL_ORDER]
    gap = np.mean(wes) - np.mean(afr)
    axs[0].annotate("", xy=(3, np.mean(wes)), xytext=(3, np.mean(afr)),
                    arrowprops=dict(arrowstyle="<->", color="#444", lw=1.0))
    axs[0].text(2.78, np.mean([np.mean(wes), np.mean(afr)]), f"gap\n{gap:.2f}",
                fontsize=BASE_FS - 1.5, color="#444", ha="right", va="center")
    panel_label(axs[0], "a")

    # b: task x stratum accuracy heatmap — SEQUENTIAL viridis (fix: was RdYlGn)
    tasks = ["correct_bioactivity", "correct_food", "correct_cultural", "correct_bioavail"]
    tlab = ["bioact.", "food", "cultural", "bioavail"]
    M = np.array([[df[(df.stratum == s)][t].mean() for t in tasks] for s in STRATUM_ORDER])
    im = axs[1].imshow(M, cmap=ACC_CMAP, vmin=0.2, vmax=1.0, aspect="auto")
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels(tlab)
    axs[1].set_yticks(range(4)); axs[1].set_yticklabels(STRATUM_ORDER)
    for i in range(4):
        for j in range(4):
            v = M[i, j]
            axs[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.62 else "black", fontsize=BASE_FS - 1)
    cb = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    cb.set_label("accuracy (↑ better)", fontsize=BASE_FS - 1)
    axs[1].set_title("African cultural-context is the floor")
    panel_label(axs[1], "b")

    fig.tight_layout(w_pad=1.8)
    return save(fig, "fig06_cultural_gap.png")


# ============================================================================
# FIG 7 — failure structure by task and stratum  (100% stacked, threaded palette)
# ============================================================================
def fig7():
    df = D("bench100")
    df["stratum"] = df["stratum"].map(norm_stratum)
    tasks = [("correct_bioactivity", "Bioactivity\nclass"),
             ("correct_food", "Food\nsource"),
             ("correct_cultural", "Cultural\ncontext"),
             ("correct_bioavail", "Bio-\navailability")]
    fig, ax = plt.subplots(figsize=(COL_2 * 0.72, 2.8))
    ylabels, totals = [], []
    data = {s: [] for s in STRATUM_ORDER}
    for col, lab in tasks:
        fails = df[df[col] == False]
        tot = len(fails)
        totals.append(tot); ylabels.append(f"{lab}")
        for s in STRATUM_ORDER:
            data[s].append((fails.stratum == s).sum())
    y = np.arange(len(tasks))
    left = np.zeros(len(tasks))
    for s in STRATUM_ORDER:
        vals = np.array(data[s])
        ax.barh(y, vals, left=left, color=STRATUM[s], label=s, height=0.66,
                edgecolor="white", linewidth=0.5)
        left += vals
    for i, tot in enumerate(totals):
        ax.text(tot + 2, i, f"n={tot}", va="center", fontsize=BASE_FS - 1.5, color="#333")
    ax.set_yticks(y); ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlabel(f"failure instances (total {int(sum(totals))} across 4 tiers)")
    ax.set_title("Coverage failures concentrate in African compounds")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=BASE_FS - 1.5, handlelength=1.0, columnspacing=1.0)
    fig.tight_layout()
    return save(fig, "fig07_failure_structure.png")


# ============================================================================
# FIG 8 — grounding ablation  [a: gap by condition | b: African A->C rise]
# ============================================================================
def fig8():
    df = D("ablation")
    df["stratum"] = df["stratum"].map(norm_stratum)
    fig, axs = plt.subplots(1, 3, figsize=(COL_2 + 0.6, 2.7))

    # a: Western-African gap per condition (COND palette)
    conds = ["A", "B", "C", "D"]
    gaps = []
    for c in conds:
        sub = df[df.condition == c]
        w = sub[sub.stratum == "Western"]["overall"].mean()
        a = sub[sub.stratum == "African"]["overall"].mean()
        gaps.append(w - a)
    bars = axs[0].bar(range(4), gaps, color=[COND[c] for c in conds],
                      edgecolor="white", linewidth=0.6, width=0.68)
    axs[0].axhline(0, color="black", lw=0.7)
    axs[0].set_xticks(range(4))
    axs[0].set_xticklabels([f"{c}\n{COND_LABEL[c]}" for c in conds], fontsize=BASE_FS - 1.5)
    axs[0].set_ylabel("Western − African gap")
    axs[0].set_title("Provenance collapses gap", fontsize=BASE_FS)
    for i, g in enumerate(gaps):
        axs[0].text(i, g + 0.008, f"{g:.2f}", ha="center", fontsize=BASE_FS - 1.5)
    panel_label(axs[0], "a")

    # b: African accuracy A vs C, paired (shows the rise)
    hs = D("hardened")
    ap = hs["african_provenance_effect"]
    axs[1].bar([0, 1], [ap["A_mean"], ap["C_mean"]],
               color=[COND["A"], COND["C"]], width=0.6, edgecolor="white", linewidth=0.6)
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["A\nbaseline", "C\n+provenance"])
    axs[1].set_ylabel("African accuracy (↑ better)")
    axs[1].set_ylim(0, 1)
    axs[1].set_title("African accuracy rises", fontsize=BASE_FS)
    axs[1].annotate("", xy=(1, ap["C_mean"]), xytext=(0, ap["A_mean"]),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
    p = ap["wilcoxon_p_C_gt_A"]
    axs[1].text(0.5, max(ap["A_mean"], ap["C_mean"]) + 0.05,
                f"+{ap['C_mean']-ap['A_mean']:.2f}\nP={p:.1e}", ha="center",
                fontsize=BASE_FS - 1.5, color="#444")
    panel_label(axs[1], "b")

    # c: supervised structure probe — how much does chemistry-pretrained structure carry?
    pr = D("structure_probe")            # DataFrame: representation, task, chance, logreg_balacc
    tasks = ["cultural stratum", "bioactivity class", "bioavailability tier"]
    reps = ["ChemBERTa", "RDKit-physchem"]
    pcol = {"ChemBERTa": "#0072B2", "RDKit-physchem": "#E69F00"}
    x = np.arange(len(tasks)); w = 0.36
    for i, rep in enumerate(reps):
        sub = pr[pr.representation == rep].set_index("task").loc[tasks]
        axs[2].bar(x + (i - 0.5) * w, sub["logreg_balacc"], w, label=rep,
                   color=pcol[rep], edgecolor="white", linewidth=0.6)
    for j, t in enumerate(tasks):
        ch = pr[pr.task == t]["chance"].iloc[0]
        axs[2].hlines(ch, x[j] - 0.5, x[j] + 0.5, color="#444", linestyle="--", lw=1.0, zorder=5)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(["cultural\nstratum", "bioactivity\nclass", "bioavail.\ntier"],
                           fontsize=BASE_FS - 2)
    axs[2].set_ylabel("probe balanced acc.")
    axs[2].set_ylim(0, 0.8)
    axs[2].set_title("Structure carries gap weakly", fontsize=BASE_FS)
    axs[2].legend(frameon=False, fontsize=BASE_FS - 2.5, loc="upper right",
                  handlelength=1.0, borderaxespad=0.2)
    axs[2].text(0, pr.query("representation=='ChemBERTa' and task=='cultural stratum'")
                ["logreg_balacc"].iloc[0] + 0.02, "chance", fontsize=BASE_FS - 3,
                ha="center", color="#444")
    panel_label(axs[2], "c")

    fig.tight_layout(w_pad=1.4)
    return save(fig, "fig08_ablation.png")


# ============================================================================
# FIG 9 — task decomposition of the provenance effect (leakage control)
# ============================================================================
def fig9():
    d = D("taskdecomp")["per_task_across_6_families"]
    order = ["cultural", "food", "bioavail", "bioactivity"]
    lab = {"cultural": "cultural\ncontext", "food": "food\nsource",
           "bioavail": "bioavail.", "bioactivity": "bioactivity\nclass"}
    fig, ax = plt.subplots(figsize=(COL_2 * 0.72, 2.9))
    y = np.arange(len(order))
    gA = [d[k]["gap_A"] for k in order]
    gC = [d[k]["gap_C"] for k in order]
    # dumbbell: baseline gap -> post-provenance gap
    for i, k in enumerate(order):
        ax.plot([gA[i], gC[i]], [i, i], color="#bbb", lw=2, zorder=1)
    ax.scatter(gA, y, color="#999999", s=48, label="baseline (A)", zorder=3)
    ax.scatter(gC, y, color="#E69F00", s=48, label="+provenance (C)", zorder=3)
    ax.axvline(0, color="black", lw=0.7)
    for i, k in enumerate(order):
        p = d[k]["wilcoxon_p"]
        star = "*" if (isinstance(p, (int, float)) and p < 0.05) else "n.s."
        ax.text(max(gA[i], gC[i]) + 0.03, i, f"P={p:.3f} {star}" if isinstance(p, (int, float)) else "",
                va="center", fontsize=BASE_FS - 2, color="#333")
    ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in order])
    ax.invert_yaxis()
    ax.set_xlabel("Western − African gap")
    ax.set_xlim(-0.15, 0.75)
    ax.set_title("Provenance acts on documentation tasks, not chemistry")
    ax.legend(loc="lower right", fontsize=BASE_FS - 1.5)
    fig.tight_layout()
    return save(fig, "fig09_task_decomposition.png")


# ============================================================================
# FIG 10 — cross-family dissociation  [a: gap A vs C per family | b: effects]
# ============================================================================
def fig10():
    cf = D("crossfam")
    stats = D("crossfam_stats")
    fig, axs = plt.subplots(1, 2, figsize=(COL_2, 2.8),
                            gridspec_kw={"width_ratios": [1.25, 1]})
    fams = list(cf["family"])
    short = [f.replace(" (2-tier)", "").replace("3.1:", "-").replace("2.5:", "-")
             .replace("2:", "-").replace(":", "-") for f in fams]
    y = np.arange(len(fams))
    # a: dumbbell gap_A -> gap_C per family
    for i in range(len(fams)):
        axs[0].plot([cf.gap_A[i], cf.gap_C[i]], [i, i], color="#ccc", lw=2, zorder=1)
    axs[0].scatter(cf.gap_A, y, color="#999999", s=42, label="baseline (A)", zorder=3)
    axs[0].scatter(cf.gap_C, y, color="#E69F00", s=42, label="+provenance (C)", zorder=3)
    axs[0].axvline(0, color="black", lw=0.7)
    axs[0].set_yticks(y); axs[0].set_yticklabels(short, fontsize=BASE_FS - 1.5)
    axs[0].invert_yaxis()
    axs[0].set_xlabel("Western − African gap")
    axs[0].set_title("Every family: provenance shrinks the gap")
    axs[0].legend(loc="lower right", fontsize=BASE_FS - 2)
    panel_label(axs[0], "a")

    # b: structure vs provenance effect, mean over families
    meta = stats["meta"]
    eff = [meta["structure_effect_mean"], meta["provenance_effect_mean"]]
    pvals = [meta["structure_wilcoxon_p"], meta["provenance_wilcoxon_p"]]
    axs[1].bar([0, 1], eff, color=["#56B4E9", "#E69F00"], width=0.6,
               edgecolor="white", linewidth=0.6)
    axs[1].axhline(0, color="black", lw=0.7)
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["+structure", "+provenance"])
    axs[1].set_ylabel("mean Δ gap (6 families)")
    axs[1].set_ylim(-0.17, 0.03)
    for i, (e, p) in enumerate(zip(eff, pvals)):
        axs[1].text(i, 0.006, f"P={p:.3f}", ha="center", fontsize=BASE_FS - 1.5)
    axs[1].set_title("Structure: no effect", fontsize=TITLE_FS, pad=8)
    panel_label(axs[1], "b")

    fig.tight_layout(w_pad=1.8)
    return save(fig, "fig10_crossfamily.png")


# ============================================================================
# FIG 11 — 100 vs 200 stability + scale trend
# ============================================================================
def fig11():
    c = D("comparison")
    pf = c["per_family"]
    fams = list(pf.keys())
    short = [f.replace("3.1:", "-").replace("2.5:", "-").replace("2:", "-").replace(":", "-") for f in fams]
    fig, axs = plt.subplots(1, 2, figsize=(COL_2, 2.6),
                            gridspec_kw={"width_ratios": [1.2, 1]})
    x = np.arange(len(fams))
    g100 = [pf[f]["gap_A_100"] for f in fams]
    g200 = [pf[f]["gap_A_200"] for f in fams]
    w = 0.36
    axs[0].bar(x - w / 2, g100, w, color="#b0b0b0", label="100-set", edgecolor="white", linewidth=0.4)
    axs[0].bar(x + w / 2, g200, w, color="#08519c", label="200-set", edgecolor="white", linewidth=0.4)
    axs[0].set_xticks(x); axs[0].set_xticklabels(short, rotation=35, ha="right", fontsize=BASE_FS - 1.5)
    axs[0].set_ylabel("baseline Western−African gap")
    axs[0].set_title("Gap stable when compound set doubles")
    axs[0].legend(loc="upper right", fontsize=BASE_FS - 1.5)
    panel_label(axs[0], "a")

    # b: llama 8B vs 70B scale
    tr = D("crossfam_stats")["llama_scale_trend"]
    axs[1].plot([0, 1], [tr["overall_8b"], tr["overall_70b"]], "o-", color="#08306b",
                label="overall acc", lw=1.6, ms=5)
    axs[1].plot([0, 1], [tr["baseline_gap_8b"], tr["baseline_gap_70b"]], "s--",
                color="#CC79A7", label="W−African gap", lw=1.6, ms=5)
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["Llama-8B", "Llama-70B"])
    axs[1].set_ylim(0, 0.75)
    axs[1].set_title("Scale lifts accuracy, not the gap")
    axs[1].legend(loc="center right", fontsize=BASE_FS - 2)
    panel_label(axs[1], "b")

    fig.tight_layout(w_pad=1.8)
    return save(fig, "fig11_stability_scale.png")


# ============================================================================
# FIG 4 (NEW) — ALLUVIAL: stratum -> bioactivity class -> bioavailability tier
#   A dataset-composition flow that shows how compounds distribute jointly,
#   replacing three loosely-related bar panels with one integrated view.
# ============================================================================
def _draw_alluvial(ax, df):
    # collapse rare bioactivity classes into "other" for legibility
    top = df["bioactivity_vocab"].value_counts()
    keep = list(top[top >= 8].index)
    df["bio"] = df["bioactivity_vocab"].where(df["bioactivity_vocab"].isin(keep), "other")
    bio_order = [b for b in top.index if b in keep] + ["other"]
    tiers = ["low", "moderate", "high"]

    # node columns: 0 = stratum, 1 = bioactivity, 2 = bioavailability
    cols = [STRATUM_ORDER, bio_order, tiers]
    colx = [0, 1, 2]
    gap = 0.6           # vertical gap between nodes (in "compound" units)
    N = len(df)

    # compute node sizes + y positions
    node_y = {}
    node_h = {}
    for ci, items in enumerate(cols):
        sizes = []
        for it in items:
            if ci == 0: n = (df.stratum == it).sum()
            elif ci == 1: n = (df.bio == it).sum()
            else: n = (df.bioavailability == it).sum()
            sizes.append(n)
        total = sum(sizes) + gap * (len(items) - 1)
        y = total
        for it, s in zip(items, sizes):
            node_h[(ci, it)] = s
            node_y[(ci, it)] = (y - s, y)   # (bottom, top)
            y -= (s + gap)

    node_w = 0.16
    # colour flows by stratum (source)
    strat_col = STRATUM

    def draw_flows(ci, key_from, key_to):
        # cursor positions along the right edge of left nodes and left edge of right nodes
        left_cursor = {it: node_y[(ci, it)][1] for it in cols[ci]}
        right_cursor = {it: node_y[(ci + 1, it)][1] for it in cols[ci + 1]}
        # order flows by stratum for stable stacking
        for a in cols[ci]:
            for b in cols[ci + 1]:
                if ci == 0:
                    sub = df[(df.stratum == a) & (df.bio == b)]
                    scol = strat_col[a]
                else:
                    sub = df[(df.bio == a) & (df.bioavailability == b)]
                    # colour second stage by the stratum composition -> use grey blend
                    scol = "#9e9e9e"
                n = len(sub)
                if n == 0:
                    continue
                y0l = left_cursor[a]; y0r = right_cursor[b]
                left_cursor[a] -= n; right_cursor[b] -= n
                x0 = colx[ci] + node_w / 2
                x1 = colx[ci + 1] - node_w / 2
                # bezier ribbon
                verts_top = [(x0, y0l), ((x0 + x1) / 2, y0l), ((x0 + x1) / 2, y0r), (x1, y0r)]
                verts_bot = [(x1, y0r - n), ((x0 + x1) / 2, y0r - n),
                             ((x0 + x1) / 2, y0l - n), (x0, y0l - n)]
                codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                         Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
                path = Path(verts_top + verts_bot + [(x0, y0l)], codes)
                ax.add_patch(PathPatch(path, facecolor=scol, edgecolor="none", alpha=0.45))

    # rescale x so columns are spaced out
    colx = [0, 2.4, 4.8]
    draw_flows(0, None, None)
    draw_flows(1, None, None)

    # draw nodes
    for ci, items in enumerate(cols):
        for it in items:
            b, t = node_y[(ci, it)]
            col = STRATUM[it] if ci == 0 else ("#4c72b0" if ci == 1 else
                  {"low": "#d9d9d9", "moderate": "#969696", "high": "#252525"}[it])
            ax.add_patch(plt.Rectangle((colx[ci] - node_w / 2, b), node_w, t - b,
                         facecolor=col, edgecolor="white", linewidth=0.6))
            lab = f"{it} ({node_h[(ci, it)]})"
            txtcol = "black"
            ax.text(colx[ci] + (node_w / 2 + 0.06 if ci < 2 else node_w / 2 + 0.06),
                    (b + t) / 2, lab, va="center", ha="left", fontsize=BASE_FS - 2.2,
                    color=txtcol)

    ax.set_xlim(-0.7, 6.0)
    ax.set_ylim(-1, max(v[1] for v in node_y.values()) + 1)
    ax.axis("off")
    for xc, title in zip(colx, ["Cultural stratum", "Bioactivity class", "Bioavailability"]):
        ax.text(xc, max(v[1] for v in node_y.values()) + 1.5, title,
                ha="center", fontsize=BASE_FS, fontweight="bold")
    return node_y


# ============================================================================
# FIG 4bis — molecular structure gallery (kept as its own figure)
#   Uses the shared STRATUM palette for the stratum labels.
# ============================================================================
def fig_gallery():
    import io
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from PIL import Image
    b = D("compounds")
    picks = {
        "Western":     ["resveratrol", "quercetin", "oleuropein"],
        "East Asian":  ["epigallocatechin gallate", "berberine", "ginsenoside Rb1"],
        "South Asian": ["curcumin", "piperine", "withaferin A"],
        "African":     ["aspalathin", "kolaviron", "6-paradol"],
    }

    def smi(n):
        r = b[b.name.str.lower() == n.lower()]
        if len(r) == 0:
            r = b[b.name.str.lower().str.contains(n.lower().split()[0])]
        return r.iloc[0].canonical_smiles if len(r) else None

    def draw(sm):
        m = Chem.MolFromSmiles(sm)
        d = rdMolDraw2D.MolDraw2DCairo(300, 220)
        o = d.drawOptions(); o.bondLineWidth = 1.4; o.padding = 0.08
        rdMolDraw2D.PrepareAndDrawMolecule(d, m)
        d.FinishDrawing()
        return Image.open(io.BytesIO(d.GetDrawingText()))

    fig, axes = plt.subplots(4, 3, figsize=(COL_2, 8.0))
    for i, (strat, names) in enumerate(picks.items()):
        for j, nm in enumerate(names):
            ax = axes[i, j]; ax.axis("off")
            sm = smi(nm)
            if sm:
                ax.imshow(draw(sm))
            ax.set_title(nm, fontsize=BASE_FS - 1, color="#222", pad=1)
        axes[i, 0].text(-0.14, 0.5, strat, transform=axes[i, 0].transAxes, rotation=90,
                        va="center", ha="center", fontsize=TITLE_FS, fontweight="bold",
                        color=STRATUM[strat])
    fig.suptitle("Representative bioactive structures across cultural strata",
                 fontsize=TITLE_FS + 0.5, fontweight="bold", y=0.997)
    fig.tight_layout(rect=[0.02, 0, 1, 0.985])
    return save(fig, "fig04_structure_gallery.png")


# ============================================================================
# RUNNER
# ============================================================================
# ============================================================================
# SCHEMATIC FIGURES 1, 2, 12 (merged from schematics.py)
# ============================================================================
def _light(hexc, f=0.80):
    r = int(hexc[1:3], 16); g = int(hexc[3:5], 16); b = int(hexc[5:7], 16)
    r = int(r + (255 - r) * f); g = int(g + (255 - g) * f); b = int(b + (255 - b) * f)
    return f"#{r:02x}{g:02x}{b:02x}"

NAVY = "#0a3d62"


# ============================================================================
# FIG 1 — graphical abstract
# ============================================================================
def fig1():
    M = D("bench100"); M["stratum"] = M["stratum"].map(norm_stratum)
    vals = [round(M[M.stratum == s].overall.mean(), 2) for s in STRATUM_ORDER]

    fig, ax = plt.subplots(figsize=(COL_2 * 1.5, 5.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.text(5.5, 5.95, "Food AI can taste, but it cannot yet reason about health",
            fontsize=13.5, fontweight="bold", ha="center", color=NAVY)
    ax.text(5.5, 5.55, "A bioactive-aware, culturally-grounded benchmark and remedy",
            fontsize=9.5, ha="center", color="#555", style="italic")

    def box(x, y, w, h, txt, fc, ec, fs=8, bold=False, tc="#222"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                     fc=fc, ec=ec, lw=1.4, zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal", zorder=3)

    ax.text(2.15, 4.95, "WHAT FOOD AI HANDLES", fontsize=8.5, fontweight="bold",
            color="#777", ha="center")
    box(0.35, 4.35, 3.6, 0.5, "✓  Flavour association  (Pan 2026, F1<0.6)", "#e8f0f7", STRATUM["Western"], 7.6)
    box(0.35, 3.75, 3.6, 0.5, "✓  Sensor / vision safety & quality", "#e8f0f7", STRATUM["Western"], 7.6)
    box(0.35, 3.15, 3.6, 0.5, "✓  Nutrition & recipe text", "#e8f0f7", STRATUM["Western"], 7.6)
    box(0.35, 2.35, 3.6, 0.62, "✗  BIOACTIVES\n(the compounds that affect health)",
        _light(STRATUM["African"], 0.75), STRATUM["African"], 8, bold=True, tc="#7a2f5c")
    ax.add_patch(FancyArrowPatch((4.1, 3.15), (5.15, 3.15), arrowstyle="-|>",
                 mutation_scale=18, color="#333", lw=2))
    ax.text(4.62, 3.42, "we test", fontsize=6.8, color="#555", ha="center", style="italic")

    ax.text(7.05, 4.95, "WHAT WE FIND", fontsize=8.5, fontweight="bold", color="#777", ha="center")
    box(5.2, 3.95, 3.7, 0.9, "100 bioactives × 4 LLMs\n4 health-relevant tasks", "#f4f4f4", "#888", 8, bold=True)

    axins = fig.add_axes([0.495, 0.40, 0.29, 0.185])
    axins.bar(range(4), vals, color=[STRATUM[s] for s in STRATUM_ORDER], width=0.7, edgecolor="white")
    axins.set_ylim(0, 1); axins.set_xticks(range(4))
    axins.set_xticklabels(["West", "E.Asia", "S.Asia", "Africa"], fontsize=6.2)
    axins.set_yticks([0, 0.5, 1]); axins.tick_params(labelsize=6)
    for i, v in enumerate(vals):
        axins.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6, fontweight="bold")
    axins.spines[["top", "right"]].set_visible(False)
    axins.set_title("accuracy falls with cultural distance", fontsize=6.6, pad=2)

    box(5.2, 1.35, 3.7, 0.72, "Western → African gap: 23 pts (P<10⁻¹¹)\nscale narrows it, does NOT close it",
        "#fff4e6", STRATUM["South Asian"], 7.8, bold=True, tc="#b9770e")
    ax.add_patch(FancyArrowPatch((9.0, 3.0), (9.5, 3.0), arrowstyle="-|>",
                 mutation_scale=16, color="#333", lw=2))
    ax.text(10.2, 4.95, "THE FIX", fontsize=8.5, fontweight="bold", color="#777", ha="center")
    # provenance leads (validated); graph encoder demoted to untested hypothesis
    box(9.55, 3.5, 1.3, 1.2, "+ cultural\nprovenance\n→ gap −73%\n(validated)",
        _light(STRATUM["East Asian"], 0.72), STRATUM["East Asian"], 7.2, bold=True, tc="#145a32")
    box(9.55, 2.05, 1.3, 1.05, "graph\nencoder\n(untested\nhypothesis)", "#eeeeee", "#999", 7.0, tc="#666")
    ax.text(5.5, 0.85, "Three linked gaps:  bioactive absence  ·  structure-blindness  ·  compound-level cultural blindness",
            fontsize=8.2, ha="center", color=NAVY, fontweight="bold")
    ax.plot([0.4, 10.85], [1.05, 1.05], color="#ddd", lw=0.8)
    return save(fig, "fig01_graphical_abstract.png")


# ============================================================================
# FIG 12 — proposed architecture (provenance = validated, structure = hypothesis)
# ============================================================================
def fig12():
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
    return save(fig, "fig12_architecture.png")


# ============================================================================
# FIG 2 — representation ladder (rebuilt clean, threaded palette accents)
# ============================================================================
def fig2():
    fig, ax = plt.subplots(figsize=(COL_2, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.text(5, 5.95, "The representation ladder for food bioactives", fontsize=TITLE_FS + 1,
            fontweight="bold", ha="center", color=NAVY)
    rungs = [
        ("Compound name", "“curcumin”", "text token — what every food LLM uses", "#f2f2f2", "#999"),
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


FIGS = {
    1: fig1, 2: fig2,
    3: fig3, 4: fig_gallery, 5: fig5, 6: fig6, 7: fig7,
    8: fig8, 9: fig9, 10: fig10, 11: fig11, 12: fig12,
}


if __name__ == "__main__":
    setup()
    which = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else sorted(FIGS)
    for n in which:
        if n in FIGS:
            print(f"Fig {n}:")
            FIGS[n]()


