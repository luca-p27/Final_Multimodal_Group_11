import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.normpath(os.path.join(_HERE, "..", ".."))

FIG_MAIN = os.path.join(ROOT, "figures", "main")
FIG_SUPP = os.path.join(ROOT, "figures", "supplementary")
os.makedirs(FIG_MAIN, exist_ok=True)
os.makedirs(FIG_SUPP, exist_ok=True)

# Paths to input data and model outputs
OCCURRENCE  = os.path.join(ROOT, "input", "CrypticBio-Common_continent.tsv")
CRYPTIC_TSV = os.path.join(ROOT, "input", "small_lookup_cryptic_group.tsv")
BASE = os.path.join(ROOT, "output", "downstream/")

MODELS = {
    "Baseline":(f"{BASE}Baseline/per_class_metrics.csv", f"{BASE}Baseline/test_predictions.csv", "Baseline"),
    "Raw (E)": (f"{BASE}Raw/per_class_metrics_raw_early.csv", f"{BASE}Raw/test_predictions_raw_early.csv", "Raw"),
    "Raw (L)":(f"{BASE}Raw/per_class_metrics_raw_late.csv", f"{BASE}Raw/test_predictions_raw_late.csv", "Raw"),
    "Wrap (E)":(f"{BASE}Wrap/per_class_metrics_wrap_early.csv", f"{BASE}Wrap/test_predictions_wrap_early.csv", "Wrap"),
    "Wrap (L)":(f"{BASE}Wrap/per_class_metrics_wrap_late.csv", f"{BASE}Wrap/test_predictions_wrap_late.csv", "Wrap"),
    "SH (E)": (f"{BASE}SH/per_class_metrics_sh_early.csv", f"{BASE}SH/test_predictions_sh_early.csv", "SH"),
    "SH (L)": (f"{BASE}SH/per_class_metrics_sh_late.csv", f"{BASE}SH/test_predictions_sh_late.csv","SH"),
    "Hex (E)": (f"{BASE}Hex/per_class_metrics_hex_early.csv", f"{BASE}Hex/test_predictions_hex_early.csv", "Hex"),
    "Hex (L)": (f"{BASE}Hex/per_class_metrics_hex_late.csv", f"{BASE}Hex/test_predictions_hex_late.csv", "Hex"),
    "Geo_both (E)":(f"{BASE}both/per_class_metrics_geo_label_early.csv", f"{BASE}both/test_predictions_geo_label_early.csv", "Geo (both)"),
    "Geo_both (L)": (f"{BASE}both/per_class_metrics_geo_label_late.csv", f"{BASE}both/test_predictions_geo_label_late.csv", "Geo (both)"),
    "Geo_Country (E)": (f"{BASE}country/per_class_metrics_geo_label_early.csv", f"{BASE}country/test_predictions_geo_label_early.csv", "Geo (country)"),
    "Geo_Country (L)": (f"{BASE}country/per_class_metrics_geo_label_late.csv", f"{BASE}country/test_predictions_geo_label_late.csv", "Geo (country)"),
    "Geo_Continent (E)": (f"{BASE}continent/per_class_metrics_geo_label_early.csv", f"{BASE}continent/test_predictions_geo_label_early.csv", "Geo (continent)"),
    "Geo_Continent (L)": (f"{BASE}continent/per_class_metrics_geo_label_late.csv", f"{BASE}continent/test_predictions_geo_label_late.csv", "Geo (continent)"), }

BEST_GEO = "SH (L)"
LATE_MODELS = ["Baseline", "Raw (L)", "Wrap (L)", "SH (L)", "Hex (L)", "Geo_both (L)", "Geo_Country (L)", "Geo_Continent (L)"]

# Visual choices

plt.rcParams.update({
    "font.family":"sans-serif",
    "font.sans-serif":  ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

TAXON_COLORS = {
    "Aves": "#2FE7BC",
    "Insecta": "#FABA3A",
    "Arachnida":"#4585FB",
    "Squamata":"#7A0402",
    "Gastropoda":"#A0FC39",
    "Magnoliopsida":"#E4460A",
    "Agaricomycetes": "#30123B",
}
MODEL_COLORS = {
    "Baseline": "#000000", 
    "Geo (both)": "#117733",
    "Geo (country)": "#88CCEE", 
    "Geo (continent)": "#44AA99",
    "Raw":"#DDCC77", 
    "Wrap":"#AA4499",
    "SH":"#CC6677", 
    "Hex":"#882255",
}
LATE_COL = {
    "Baseline": MODEL_COLORS["Baseline"], 
    "Raw (L)":MODEL_COLORS["Raw"],      
    "Wrap (L)": MODEL_COLORS["Wrap"],
    "SH (L)": MODEL_COLORS["SH"],       
    "Hex (L)":MODEL_COLORS["Hex"],
    "Geo_both (L)": MODEL_COLORS["Geo (both)"],
    "Geo_Country (L)": MODEL_COLORS["Geo (country)"],
    "Geo_Continent (L)": MODEL_COLORS["Geo (continent)"],
}

def load_metrics(path):
    df = pd.read_csv(path, index_col=0)
    return df[~df.index.str.contains("accuracy|macro avg|weighted avg", na=False)]

def load_predictions(path):
    df = pd.read_csv(path)
    df["correct"] = df["True_Species"] == df["Predicted_Species"]
    return df

def load_occurrence():
    return pd.read_csv(OCCURRENCE, sep="\t")

def load_cryptic_pairs():
    """Return set of (a, b) pairs in both directions."""
    pairs = set()
    if not os.path.exists(CRYPTIC_TSV):
        return pairs
    with open(CRYPTIC_TSV, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 1:
                anchor = parts[0].strip()
                siblings = [s.strip() for s in parts[1].split(",") if s.strip()] \
                           if len(parts) > 1 and parts[1].strip() else []
                for sib in siblings:
                    pairs.add((anchor, sib))
                    pairs.add((sib, anchor))
    return pairs

def build_cryptic_group_sizes(species_set):
    """
    For each species in species_set, count how many of its cryptic siblings
    also appear in species_set.  Range 1–N gives meaningful visual variation.
    """
    siblings_in_dataset = {sp: set() for sp in species_set}
    if not os.path.exists(CRYPTIC_TSV):
        return {sp: 1 for sp in species_set}

    with open(CRYPTIC_TSV, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 1:
                continue
            anchor = parts[0].strip()
            siblings = [s.strip() for s in parts[1].split(",") if s.strip()] \
                       if len(parts) > 1 and parts[1].strip() else []
            if anchor in species_set:
                for sib in siblings:
                    if sib in species_set:
                        siblings_in_dataset[anchor].add(sib)
                        siblings_in_dataset[sib].add(anchor)

    return {sp: max(1, len(sibs)) for sp, sibs in siblings_in_dataset.items()}

def available_models():
    return {n: (m, p, g) for n, (m, p, g) in MODELS.items()
            if os.path.exists(m) and os.path.exists(p)}

def build_results_df(avail):
    rows = []
    for name, (m_path, _, group) in avail.items():
        raw = pd.read_csv(m_path, index_col=0)
        sp = raw[~raw.index.str.contains("accuracy|macro avg|weighted avg", na=False)]
        acc = float(raw.loc["accuracy", "f1-score"]) \
              if "accuracy" in raw.index else sp["f1-score"].mean()
        rows.append({"model": name, "group": group, "accuracy": acc, "macro_f1": sp["f1-score"].mean(), "macro_p":  sp["precision"].mean(), "macro_r":  sp["recall"].mean()})
    return pd.DataFrame(rows)

def build_gains_df(avail, occ):
    if "Baseline" not in avail or BEST_GEO not in avail:
        return pd.DataFrame()
    bm = load_metrics(avail["Baseline"][0])
    gm = load_metrics(avail[BEST_GEO][0])
    sp_taxon = occ.groupby("scientificName")["class"].first()
    sp_nconts = occ.groupby("scientificName")["continent"].nunique()
    common = bm.index.intersection(gm.index)
    df = pd.DataFrame({ "species": common, "f1_base": bm.loc[common, "f1-score"].values, "f1_geo": gm.loc[common, "f1-score"].values, "f1_gain": (gm.loc[common, "f1-score"] - bm.loc[common, "f1-score"]).values, })
    df["taxon"] = df["species"].map(sp_taxon).fillna("Unknown")
    df["n_conts"] = df["species"].map(sp_nconts).fillna(1)

    group_sizes = build_cryptic_group_sizes(set(common))
    df["cryptic_group_size"] = df["species"].map(group_sizes).fillna(1)
    return df

# Helper for plot (kernal-density)
def _draw_stacked_kde(ax_top, ax_bot, model_list, avail, xr, highlight=None, show_legend=True):

    for mname in model_list:
        if mname not in avail:
            continue
        preds = load_predictions(avail[mname][1])
        c_vals = preds.loc[ preds["correct"], "Confidence"].dropna().values
        e_vals = preds.loc[~preds["correct"], "Confidence"].dropna().values
        if len(c_vals) < 5 or len(e_vals) < 5:
            continue

        col = LATE_COL.get(mname, "#888")
        lw = 1.9 if mname == highlight else 1.0
        ls = "-" if mname == highlight else "--"
        alpha = 0.90 if mname == highlight else 0.60

        kde_c = gaussian_kde(c_vals, bw_method=0.10)
        kde_e = gaussian_kde(e_vals, bw_method=0.10)

        ax_top.plot(xr, kde_c(xr), color=col, lw=lw, ls=ls, alpha=alpha, label=mname)
        ax_bot.plot(xr, kde_e(xr), color=col, lw=lw, ls=ls, alpha=alpha, label=mname)

        if mname == highlight:
            ax_top.fill_between(xr, kde_c(xr), alpha=0.10, color=col)
            ax_bot.fill_between(xr, kde_e(xr), alpha=0.10, color=col)

    for ax, tag in [(ax_top, "Correct"), (ax_bot, "Incorrect")]:
        ax.set_xlim(0, 1)
        ax.set_ylabel("Density", fontsize=7)
        ax.grid(linewidth=0.3, alpha=0.35)
        ax.text(0.97, 0.93, tag, transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#444", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ccc", lw=0.5, alpha=0.9))

    ax_bot.set_xlabel("Softmax confidence", fontsize=8)
    ax_top.set_xticklabels([])

    if show_legend:
        ax_top.legend(fontsize=6, loc="upper left", framealpha=0.75, handlelength=1.6, labelspacing=0.25, ncol=1)

# FIG 1 
def main_fig1_overview(results_df, out="figures/main/fig1_model_overview.png"):
    df = results_df.sort_values("macro_f1", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    n, h, off = len(df), 0.28, 0.15
    y = np.arange(n)
    colors = [MODEL_COLORS.get(g, "#888") for g in df["group"]]

    ax.barh(y + off, df["accuracy"], height=h, color=colors, alpha=0.50, edgecolor="white", linewidth=0.3)
    bars_f1 = ax.barh(y - off, df["macro_f1"], height=h, color=colors, alpha=0.92, edgecolor="white", linewidth=0.3)

    for bar, val in zip(bars_f1, df["macro_f1"]):
        ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=6)

    bl = df[df["model"] == "Baseline"]
    if not bl.empty:
        ax.axvline(bl["accuracy"].values[0], color="#555", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axvline(bl["macro_f1"].values[0], color="#555", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=7.5)
    ax.set_xlim(0, df[["accuracy", "macro_f1"]].max().max() + 0.09)
    ax.set_xlabel("Score", fontsize=8)
    ax.set_title("(A)  Model performance overview", fontsize=8.5, pad=5)
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)

    handles_col = [mpatches.Patch(color=c, label=k) 
                   for k, c in MODEL_COLORS.items() if k in df["group"].values]
    handles_sty = [mpatches.Patch(facecolor="#888", alpha=0.50, label="Accuracy"), mpatches.Patch(facecolor="#888", alpha=0.92, label="Macro-F1")]
    leg1 = ax.legend(handles=handles_col, loc="lower right", fontsize=6, framealpha=0.75, title="Encoding", title_fontsize=6.5, bbox_to_anchor=(1.05, 0))
    ax.add_artist(leg1)
    ax.legend(handles=handles_sty, loc="center right", fontsize=6, framealpha=0.75, bbox_to_anchor=(1.025, 0.5))

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f" {out}")

# FIG 2

def main_fig2_calibration(avail, out="figures/main/fig2_calibration.png"):

    xr = np.linspace(0, 1, 400)

    model_data = {}  
    for mname in LATE_MODELS: #Collect last fusion model info
        if mname not in avail:
            continue
        preds  = load_predictions(avail[mname][1])
        c_vals = preds.loc[ preds["correct"], "Confidence"].dropna().values
        e_vals = preds.loc[~preds["correct"], "Confidence"].dropna().values
        if len(c_vals) > 5 and len(e_vals) > 5:
            model_data[mname] = (c_vals, e_vals)

    fig = plt.figure(figsize=(13, 3.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.08, wspace=0.32, height_ratios=[1, 1], width_ratios=[1, 1])
    ax_top = fig.add_subplot(gs[0, 0])   # correct KDE
    ax_bot = fig.add_subplot(gs[1, 0])   # error KDE
    ax_gap = fig.add_subplot(gs[:, 1])   # gap bar 

    gap_rows = []
    for mname, (c_vals, e_vals) in model_data.items():
        col = LATE_COL.get(mname, "#888")
        is_hi = (mname == BEST_GEO)
        lw = 2.0 if is_hi else 1.0
        ls = "-"  if is_hi else "--"
        alpha = 0.90 if is_hi else 0.60

        kde_c = gaussian_kde(c_vals, bw_method=0.10)
        kde_e = gaussian_kde(e_vals, bw_method=0.10)

        ax_top.plot(xr, kde_c(xr), color=col, lw=lw, ls=ls, alpha=alpha, label=mname, zorder=3)
        ax_bot.plot(xr, kde_e(xr), color=col, lw=lw, ls=ls, alpha=alpha, zorder=3)

        if is_hi:
            ax_top.fill_between(xr, kde_c(xr), alpha=0.12, color=col)
            ax_bot.fill_between(xr, kde_e(xr), alpha=0.12, color=col)

        gap_rows.append({ "model": mname, "gap":   c_vals.mean() - e_vals.mean(), "color": col, })

    for ax, tag in [(ax_top, "Correct"), (ax_bot, "Incorrect")]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 5)
        ax.set_yticks(np.arange(0, 5.5, 1))
        ax.set_ylabel("Density", fontsize=8)
        ax.grid(linewidth=0.3, alpha=0.35)
        ax.text(0.97, 0.92, tag, transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#333", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb", lw=0.5, alpha=0.92))

    ax_bot.set_xlabel("Softmax confidence", fontsize=8)
    ax_top.set_xticklabels([])
    ax_top.set_title("(A) ", fontsize=9, pad=5)
    ax_top.legend(fontsize=6.5, loc="upper left", framealpha=0.80, handlelength=1.6, labelspacing=0.25, ncol=1)

    #Left side
    gap_df   = pd.DataFrame(gap_rows).sort_values("gap", ascending=True)
    bl_gap   = gap_df.loc[gap_df["model"] == "Baseline", "gap"]
    bl_gap   = bl_gap.values[0] if not bl_gap.empty else None

    bars = ax_gap.barh(gap_df["model"], gap_df["gap"], color=gap_df["color"].tolist(), height=0.60, edgecolor="white", linewidth=0.3)

    for bar, (_, row) in zip(bars, gap_df.iterrows()):
        fw = "bold" if row["model"] == BEST_GEO else "normal"
        ax_gap.text(row["gap"] + 0.004, bar.get_y() + bar.get_height() / 2, f"{row['gap']:.3f}", va="center", fontsize=7.5, fontweight=fw)
        if row["model"] == BEST_GEO:
            bar.set_edgecolor(MODEL_COLORS["SH"])
            bar.set_linewidth(1.5)

    if bl_gap is not None:
        ax_gap.axvline(bl_gap, color="#888", linestyle="--", linewidth=0.9, alpha=0.7, zorder=0)

    ax_gap.set_xlabel("Confidence gap  (μ$_{correct}$ − μ$_{error}$)", fontsize=8)
    ax_gap.set_title("(B)",fontsize=9, pad=5)
    ax_gap.set_xlim(0, gap_df["gap"].max() + 0.08)
    ax_gap.grid(axis="x", linewidth=0.3, alpha=0.4)
    ax_gap.tick_params(axis="y", labelsize=7.5)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"{out}")

# FIG 3 — cryptic analysis: taxon F1 gain and  per-species scatter 

def main_fig3_cryptic(gains_df, out="figures/main/fig3_cryptic_analysis.png"):

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    fig.subplots_adjust(wspace=0.38)
    rng = np.random.default_rng(42)

    # Panel A: Cleveland dot plot 
    ax = axes[0]
    taxon_order = (gains_df.groupby("taxon")["f1_gain"].median().sort_values(ascending=True).index.tolist())

    for i, tax in enumerate(taxon_order):
        sub = gains_df.loc[gains_df["taxon"] == tax, "f1_gain"].values
        if len(sub) == 0: continue
        col = TAXON_COLORS.get(tax, "#888")
        med = np.median(sub)
        q1, q3 = np.percentile(sub, 25), np.percentile(sub, 75)

        ax.plot([sub.min(), sub.max()], [i, i], color=col, lw=0.9, alpha=0.40, solid_capstyle="round", zorder=2)
        ax.plot([q1, q3], [i, i], color=col, lw=5, alpha=0.28, solid_capstyle="round", zorder=2)
        jit = rng.uniform(-0.20, 0.20, len(sub))
        ax.scatter(sub, i + jit, color=col, s=14, alpha=0.60, linewidths=0, zorder=3)
        ax.scatter([med], [i], color=col, s=55, zorder=5, edgecolors="white", linewidths=1.0)

    ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.55)
    ax.set_yticks(range(len(taxon_order)))
    ax.set_yticklabels(taxon_order, fontsize=8)
    ax.set_xlabel(f"F1 gain  ({BEST_GEO} − Baseline)", fontsize=8)
    ax.set_title(f"(A) ",fontsize=8.5, pad=4)
    ax.set_ylim(-0.6, len(taxon_order) - 0.4)
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)

    n_imp = (gains_df["f1_gain"] > 0).sum()
    pct= n_imp / len(gains_df) * 100
    ax.text(0.98, 0.02, f"{pct:.0f}% improved  (n={n_imp})",transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#2ca02c", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", lw=0.6, alpha=0.9))

    leg_els = [
        mlines.Line2D([0],[0], color="#888", lw=5, alpha=0.28, label="IQR"),
        mlines.Line2D([0],[0], color="#888", lw=0.9, alpha=0.45, label="Range"),
        mlines.Line2D([0],[0], marker="o", color="w", markerfacecolor="#888", markeredgecolor="white", markersize=7, label="Median"),
    ]
    ax.legend(handles=leg_els, fontsize=6.5, loc="center right", framealpha=0.75, handlelength=1.5)

    ax = axes[1]
    cgs = gains_df["cryptic_group_size"].values
    cgs_min, cgs_max = cgs.min(), cgs.max()
    if cgs_max > cgs_min:
        s_vals = 20 + (cgs - cgs_min) / (cgs_max - cgs_min) * 140
    else:
        s_vals = np.full(len(cgs), 60)

    for idx, (_, row) in enumerate(gains_df.iterrows()):
        col = TAXON_COLORS.get(row["taxon"], "#888")
        ax.scatter(row["f1_base"], row["f1_geo"], c=col, s=s_vals[idx], alpha=0.70, linewidths=0, zorder=3)

    ax.plot([0,1],[0,1], color="#aaa", lw=0.9, ls="--", zorder=2)
    ax.fill_between([0,1],[0,1],[1,1], alpha=0.05, color="#2ca02c")
    ax.fill_between([0,1],[0,0],[0,1], alpha=0.05, color="#d62728")
    ax.text(0.04, 0.97, "improved", fontsize=7, color="#2ca02c", transform=ax.transAxes, va="top")
    ax.text(0.96, 0.06, "worsened", fontsize=7, color="#d62728", transform=ax.transAxes, va="bottom", ha="right")

    pct2 = (gains_df["f1_geo"] > gains_df["f1_base"]).mean() * 100

    for sz_label, sz_val in [(f"min ({int(cgs_min)})", cgs_min), (f"max ({int(cgs_max)})", cgs_max)]:
        s_pt = 20 + (sz_val - cgs_min) / max(cgs_max - cgs_min, 1) * 140
        ax.scatter([], [], c="#888", s=s_pt, alpha=0.70, linewidths=0, label=f"group size {sz_label}")
    

    ax.set_xlabel("F1 — Baseline", fontsize=8)
    ax.set_ylabel(f"F1 — {BEST_GEO}", fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"(B)  Per-species F1: Baseline vs {BEST_GEO}", fontsize=8.5, pad=4)
    ax.grid(linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f" {out}")


def main_fig4_misclassification(avail, occ, out="figures/main/fig4_misclassification.png"):

    sp_taxon = occ.groupby("scientificName")["class"].first()
    pairs = load_cryptic_pairs()

    def annotate(preds):
        preds = preds.copy()
        preds["taxon"] = preds["True_Species"].map(sp_taxon)
        preds["is_cryptic_error"] = (~preds["correct"]) & preds.apply( lambda r: (r["True_Species"], r["Predicted_Species"]) in pairs, axis=1)
        preds["is_noncryptic_error"] = ((~preds["correct"]) & ~preds["is_cryptic_error"])
        return preds

    bl_pred = annotate(load_predictions(avail["Baseline"][1]))
    geo_pred = annotate(load_predictions(avail[BEST_GEO][1]))

    bl_pred["_id"] = bl_pred.index
    geo_pred["_id"] = geo_pred.index

    geo_lookup = geo_pred.set_index("_id")[["correct", "Predicted_Species"]]
    taxa_order = sorted(TAXON_COLORS.keys())

    def error_counts(preds):
        crypt  = preds[preds["is_cryptic_error"]].groupby("taxon").size()
        ncrypt = preds[preds["is_noncryptic_error"]].groupby("taxon").size()
        return (crypt.reindex(taxa_order,  fill_value=0), ncrypt.reindex(taxa_order, fill_value=0))

    bl_cry,  bl_ncy  = error_counts(bl_pred)
    geo_cry, geo_ncy = error_counts(geo_pred)
    bl_errors = bl_pred[~bl_pred["correct"]].copy()

    top_pairs = (bl_errors.groupby(["True_Species", "Predicted_Species"]).agg(bl_count=("True_Species", "count")).reset_index().sort_values("bl_count", ascending=False).head(10))

    # For every top pair, find the SH outcome for those exact samples
    fate_rows = []
    for _, row in top_pairs.iterrows():
        mask = ((bl_errors["True_Species"]    == row["True_Species"]) & (bl_errors["Predicted_Species"] == row["Predicted_Species"]))
        sample_ids = bl_errors.loc[mask, "_id"]

        sh_outcomes  = geo_lookup.loc[geo_lookup.index.isin(sample_ids)]
        n_correct = int(sh_outcomes["correct"].sum())
        n_wrong = int((~sh_outcomes["correct"]).sum()) 
        n_found = len(sh_outcomes)                       

        fate_rows.append({
            "True_Species":  row["True_Species"],
            "Predicted_Species": row["Predicted_Species"],
            "bl_count":row["bl_count"],
            "sh_correct": n_correct,
            "sh_wrong":n_wrong,
            "n_found": n_found,
        })

    top_pairs = pd.DataFrame(fate_rows)
    top_pairs["is_cryptic"] = top_pairs.apply( lambda r: (r["True_Species"], r["Predicted_Species"]) in pairs, axis=1)
    top_pairs["taxon"] = top_pairs["True_Species"].map(sp_taxon)
    top_pairs["label"] = (top_pairs["True_Species"].str.split().str[-1]+ " \n" + top_pairs["Predicted_Species"].str.split().str[-1])
    top_pairs = top_pairs.sort_values("bl_count", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.subplots_adjust(wspace=0.42)
 
    ax = axes[0]
    x, w = np.arange(len(taxa_order)), 0.35

    ax.bar(x - w/2, bl_cry,  width=w, color=[TAXON_COLORS[t] for t in taxa_order], alpha=0.85, label="Baseline — cryptic", zorder=3)
    ax.bar(x - w/2, bl_ncy, width=w, bottom=bl_cry, color=[TAXON_COLORS[t] for t in taxa_order], alpha=0.35, hatch="///", edgecolor="white", label="Baseline — non-cryptic", zorder=3)
    ax.bar(x + w/2, geo_cry,  width=w, color=[TAXON_COLORS[t] for t in taxa_order], alpha=0.85, edgecolor="#CC6677", linewidth=0.8, label=f"{BEST_GEO} — cryptic", zorder=3)
    ax.bar(x + w/2, geo_ncy, width=w, bottom=geo_cry, color=[TAXON_COLORS[t] for t in taxa_order], alpha=0.35, hatch="///", edgecolor="#CC6677", linewidth=0.8, label=f"{BEST_GEO} — non-cryptic", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([t[:20] for t in taxa_order], fontsize=7, rotation=20)
    ax.set_ylabel("Number of errors", fontsize=8)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    leg_handles = [
        mpatches.Patch(facecolor="#888", alpha=0.85, label="Cryptic errors"),
        mpatches.Patch(facecolor="#888", alpha=0.35, hatch="///", edgecolor="white", label="Non-cryptic errors"),
        mpatches.Patch(facecolor="none", edgecolor="#000000", lw=1.2, label="Baseline"),
        mpatches.Patch(facecolor="none", edgecolor="#CC6677", lw=1.2, label=BEST_GEO),
    ]
    ax.legend(handles=leg_handles, fontsize=6, loc="upper right", framealpha=0.8, ncol=2)

    ax = axes[1]
    y = np.arange(len(top_pairs))

    BAR_H = 0.30 
    GREEN = "#5fcde4"
    RED = "#ac3232"
    GREY_LINE = "#cccccc"

    for i, row in top_pairs.iterrows():
        col = TAXON_COLORS.get(row["taxon"], "#888")
        marker = "D" if row["is_cryptic"] else "o"

        # baseline dot (right anchor)
        ax.scatter(row["bl_count"], i, color=col, s=60, marker=marker, zorder=5, edgecolors="#333" if row["is_cryptic"] else col, linewidths=0.8)
        ax.barh(i, row["sh_correct"], height=BAR_H, left=0, color=GREEN, alpha=0.80, zorder=3)
        ax.barh(i, row["sh_wrong"], height=BAR_H, left=row["sh_correct"], color=RED, alpha=0.80, zorder=3)

        bar_right = row["sh_correct"] + row["sh_wrong"]
        ax.plot([bar_right, row["bl_count"]], [i, i], color=GREY_LINE, lw=1.0, zorder=1, ls="--")
        pct = row["sh_correct"] / row["bl_count"] * 100 if row["bl_count"] else 0
        ax.text(row["bl_count"] + 0.18, i, f"   {pct:.0f}% correct", va="center", fontsize=6.0, color="#444")

    ax.axvline(0, color="#aaa", lw=0.6, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(top_pairs["label"].values, fontsize=6.5)
    ax.set_xlabel("Number of errors (test set)", fontsize=8)
    ax.set_title(f"(B)  Top-10 error pairs; fate under {BEST_GEO}", fontsize=8.5, pad=4)
    ax.set_xlim(-0.5, top_pairs["bl_count"].max() + 3.5)
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)

    # legend
    leg2_handles = [
        mlines.Line2D([0], [0], marker="D", color="w", markerfacecolor="#888", markeredgecolor="#333", markersize=6, label="Baseline (cryptic)"),
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", markeredgecolor="#888", markersize=6, label="Baseline (non-cryptic)"),
        mpatches.Patch(facecolor=GREEN, alpha=0.80, label=f"{BEST_GEO} (correct)"),
        mpatches.Patch(facecolor=RED,   alpha=0.80, label=f"{BEST_GEO} (still wrong)"),
    ]
    ax.legend(handles=leg2_handles, fontsize=6.0, loc="lower right", framealpha=0.85, labelspacing=0.35)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f" {out}")


# SUPP FIG 1

def supp_fig1_overall(results_df, out="figures/supplementary/supp_fig1_overall.png"):
    df = results_df.sort_values("macro_f1", ascending=True)
    bl_acc = results_df.loc[results_df["model"]=="Baseline","accuracy"]
    bl_f1 = results_df.loc[results_df["model"]=="Baseline","macro_f1"]
    bl_acc = bl_acc.values[0] if not bl_acc.empty else None
    bl_f1 = bl_f1.values[0]  if not bl_f1.empty  else None

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    fig.subplots_adjust(wspace=0.04)

    for ax, metric, title, bval in zip( axes, ["accuracy","macro_f1"], ["(A)  Top-1 Accuracy","(B)  Macro-F1"], [bl_acc, bl_f1]):

        colors = [MODEL_COLORS.get(g,"#888") for g in df["group"]]
        bars = ax.barh(df["model"], df[metric], color=colors, height=0.62, edgecolor="white", linewidth=0.3)
        for bar, val, grp in zip(bars, df[metric], df["group"]):
            fw = "bold" if grp == "SH" else "normal"
            ax.text(val+0.005, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", fontsize=6.5, fontweight=fw)
        if bval is not None:
            ax.axvline(bval, color="black", ls="--", lw=0.8, alpha=0.5, zorder=0)
        ax.set_xlim(0, df[metric].max()+0.07)
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel(title.split("  ")[1], fontsize=8)
        ax.grid(axis="x", linewidth=0.3, alpha=0.4)
        if ax != axes[0]:
            ax.set_yticklabels([])
        else:
            ax.tick_params(axis="y", labelsize=7.5)

    handles = [mpatches.Patch(color=c, label=k) 
                for k, c in MODEL_COLORS.items() if k in df["group"].values]
    handles += [mlines.Line2D([0],[0], color="black", ls="--", lw=0.9, label="Baseline")]
    axes[1].legend(handles=handles, loc="lower right", fontsize=6.5, framealpha=0.75)
    fig.suptitle("Supplementary Fig. 1 — Full model performance", fontsize=8.5, y=1.01)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f" {out}")

# SUPP FIG 2

def supp_fig2_range(gains_df, out="figures/supplementary/supp_fig2_range_breadth.png"):
    if gains_df.empty:
        print(f"  [skip] {out}"); return
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    rng = np.random.default_rng(0)
    for tax, sub in gains_df.groupby("taxon"):
        sub = sub.dropna(subset=["n_conts","f1_gain"])
        jitter = rng.uniform(-0.10, 0.10, len(sub))
        ax.scatter(sub["n_conts"] + jitter, sub["f1_gain"], c=TAXON_COLORS.get(tax,"#888"), s=22, alpha=0.65, linewidths=0, label=tax, zorder=3)
    valid = gains_df.dropna(subset=["n_conts","f1_gain"])
    if len(valid) > 5:
        z = np.polyfit(valid["n_conts"], valid["f1_gain"], 1)
        xr = np.linspace(valid["n_conts"].min(), valid["n_conts"].max(), 60)
        ax.plot(xr, np.polyval(z,xr), color="#333", lw=1.2, ls="--", alpha=0.7, zorder=4, label="OLS trend")
    ax.axhline(0, color="black", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("Range breadth (no. continents observed)", fontsize=8)
    ax.set_ylabel(f"F1 gain  ({BEST_GEO} − Baseline)", fontsize=8)
    ax.set_title("Supplementary Fig. 2 — Range breadth vs. F1 gain", fontsize=9)
    ax.grid(linewidth=0.3, alpha=0.4)
    handles = [mpatches.Patch(color=c, label=t, alpha=0.75)
                for t, c in TAXON_COLORS.items() if t in gains_df["taxon"].values]
    handles += [mlines.Line2D([0],[0], color="#333", lw=1.2, ls="--", label="OLS trend")]
    ax.legend(handles=handles, fontsize=6.5, loc="upper right", framealpha=0.75, ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f" {out}")

# SUPP FIG 3 
def supp_fig3_heatmap(avail, occ, out="figures/supplementary/supp_fig3_taxon_heatmap.png"):
    sp_taxon = occ.groupby("scientificName")["class"].first()
    taxa = [t for t in TAXON_COLORS if t in sp_taxon.values]
    rows = []
    for mname, (_, p_path, _) in avail.items():
        preds = load_predictions(p_path)
        preds["taxon"] = preds["True_Species"].map(sp_taxon)
        acc = preds.groupby("taxon")["correct"].mean()
        row = {"model": mname}
        for t in taxa:
            row[t] = acc.get(t, np.nan)
        rows.append(row)
    matrix = pd.DataFrame(rows).set_index("model")[taxa]
    order  = [m for m in MODELS if m in matrix.index]
    matrix = matrix.loc[order]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sns.heatmap(matrix.astype(float), ax=ax, annot=True, fmt=".2f", cmap="YlGn", linewidths=0.4, linecolor="#eee", cbar_kws={"label":"Top-1 Accuracy","shrink":0.75}, annot_kws={"size":7.5}, vmin=0.25, vmax=0.90)
    if BEST_GEO in matrix.index:
        rp = list(matrix.index).index(BEST_GEO)
        ax.add_patch(plt.Rectangle((0,rp), len(taxa), 1, fill=False, edgecolor="#CC6677", linewidth=2, zorder=5))
    ax.set_title(f"Supplementary Fig. 3 — Top-1 accuracy by model × taxon", fontsize=9)
    ax.tick_params(axis="x", labelsize=8, rotation=25)
    ax.tick_params(axis="y", labelsize=7.5, rotation=0)
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f" {out}")

# SUPP TABLE

def supp_table_results(avail, results_df,  out="figures/supplementary/supp_table_results.tex"):

    model_groups = {
        "Baseline": ("Baseline (img only)$^\\dagger$", "—", True),
        "Geo_Continent (E)": ("Geo (continent)", "Early", False),
        "Geo_Continent (L)": ("Geo (continent)", "Late", False),
        "Geo_Country (E)": ("Geo (country)", "Early", False),
        "Geo_Country (L)": ("Geo (country)", "Late", False),
        "Geo_both (E)": ("Geo (both)", "Early", False),
        "Geo_both (L)": ("Geo (both)", "Late", False),
        "Raw (E)": ("Raw", "Early", False),
        "Raw (L)": ("Raw", "Late", False),
        "Wrap (E)": ("Wrap", "Early", False),
        "Wrap (L)": ("Wrap", "Late", False),
        "SH (E)": ("SH", "Early", False),
        "SH (L)": ("SH", "Late", False),
        "Hex (E)": ("Hex", "Early", False),
        "Hex (L)": ("Hex", "Late", False),
    }
    
    available_models = {name: info for name, info in model_groups.items() 
                       if name in avail and name in results_df["model"].values}
    
    baseline_row = results_df[results_df["model"] == "Baseline"].iloc[0]
    baseline_acc = baseline_row["accuracy"]
    baseline_p = baseline_row["macro_p"]
    baseline_r = baseline_row["macro_r"]
    baseline_f1 = baseline_row["macro_f1"]
    best_f1 = results_df["macro_f1"].max()
    best_row = results_df[results_df["macro_f1"] == best_f1].iloc[0]
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Fusion} & \textbf{Accuracy} & \textbf{Macro-P} & \textbf{Macro-R} & \textbf{Macro-F1} \\",
        r"\midrule",
    ]

    lines.append(f"Baseline (img only)$^\\dagger$ & — & {baseline_acc:.3f} & {baseline_p:.3f} & {baseline_r:.3f} & {baseline_f1:.3f} \\\\")
    lines.append(r"\midrule")
    current_group = None
    for model_name, (display_name, fusion, is_baseline) in available_models.items():
        if model_name == "Baseline":
            continue
            
        row = results_df[results_df["model"] == model_name].iloc[0]
        acc = row["accuracy"]
        p = row["macro_p"]
        r = row["macro_r"]
        f1 = row["macro_f1"]
        
        acc_str = f"\\textbf{{{acc:.3f}}}" if acc == results_df[results_df["group"] == display_name]["accuracy"].max() else f"{acc:.3f}"
        p_str = f"\\textbf{{{p:.3f}}}" if p == results_df[results_df["group"] == display_name]["macro_p"].max() else f"{p:.3f}"
        r_str = f"\\textbf{{{r:.3f}}}" if r == results_df[results_df["group"] == display_name]["macro_r"].max() else f"{r:.3f}"
        f1_str = f"\\textbf{{{f1:.3f}}}" if f1 == results_df[results_df["group"] == display_name]["macro_f1"].max() else f"{f1:.3f}"

        if current_group != display_name and current_group is not None:
            lines.append(r"\midrule")
        
        lines.append(f"{display_name} & {fusion} & {acc_str} & {p_str} & {r_str} & {f1_str} \\\\")
        current_group = display_name

    gain_acc = best_f1 - baseline_acc
    gain_p = best_row["macro_p"] - baseline_p
    gain_r = best_row["macro_r"] - baseline_r
    gain_f1 = best_f1 - baseline_f1
    
    lines.append(r"\midrule")
    lines.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{Abs.\\ gain over baseline (best model)}}}} & \\textit{{{gain_acc:+.3f}}} & \\textit{{{gain_p:+.3f}}} & \\textit{{{gain_r:+.3f}}} & \\textit{{{gain_f1:+.3f}}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption*{ \textbf{Bold} = best per column within group. $^\dagger$ = image-only; all others use image + geographic modality.}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\end{table}")
    
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f" {out}")    

if __name__ == "__main__":

    avail = available_models()
    occ        = load_occurrence()
    results_df = build_results_df(avail)
    gains_df   = build_gains_df(avail, occ)

    if not gains_df.empty:
        pct = (gains_df["f1_gain"] > 0).mean() * 100
        print(f"  % improved under {BEST_GEO}: {pct:.1f}%")
        print(f"  Median F1 gain : {gains_df['f1_gain'].median():+.4f}")

    print("\n Main figures")
    main_fig1_overview(results_df)
    main_fig2_calibration(avail)
    main_fig3_cryptic(gains_df)
    main_fig4_misclassification(avail, occ)
    print("\n Supplementary")
    supp_fig1_overall(results_df)
    supp_fig2_range(gains_df)
    supp_fig3_heatmap(avail, occ)
    supp_table_results(avail, results_df) 
