# %%
import pandas as pd
df_all = pd.read_csv("data/all_results_v2.csv")
df_thresholds = pd.read_csv("data/thresholded_results.csv")

# %%
df_all.head()

# %%
df_thresholds.head()

# %%
import pandas as pd
# Join on 'join_key'
df_joined = pd.merge(
    df_thresholds,                      # threshold-level
    df_all.drop_duplicates("join_key"),  # experiment-level
    on="join_key",
    how="left",
    suffixes=("", "_exp")        # avoid column name collisions
)

# %%
print(df_all.columns)
print(df_thresholds.columns)

# %%
# Best (max IoU) rows per architecture for Hard/Easy
idx_hard = df_all.groupby("model.model_architecture")["metrics.iou_hard"].idxmax()
idx_easy = df_all.groupby("model.model_architecture")["metrics.iou_easy"].idxmax()

best_hard = df_all.loc[idx_hard].assign(Difficulty="Hard")
best_easy = df_all.loc[idx_easy].assign(Difficulty="Easy")




# %%
best_easy.head()

# %%
best_hard.head()

# %%
import pandas as pd

# Optional: nicer display names (edit to taste)
architecture_mapping = {
    "deeplabv3_resnet50": "DeepLabV3 (ResNet-50)",
    "fcn_resnet50": "FCN (ResNet-50)",
    "segformer": "SegFormer",
}
loss_mapping = {
    "BCEWithLogitsLoss": "BCEWithLogits",
    "JaccardLoss": "Jaccard (IoU)",
    "HybridBCEJaccardLoss": "BCE + Jaccard",
}

def _safe_get(row, col, default=None, fmt=None):
    if col in row and pd.notna(row[col]):
        return fmt(row[col]) if fmt else row[col]
    return default

def summarize_configs(df: pd.DataFrame, n=5, title=None):
    cols = set(df.columns)

    # keys we might print if present
    keys_common = [
        "join_key", "experiment", "experiment_date",
        "model.model_architecture", "model.loss",
        "model.pos_weight", "model.alpha", "model.lr",
        "dataloader.batch_size", "training.max_epochs",
    ]
    # extras found in threshold-level tables
    keys_extra = ["threshold", "config_path", "metrics_path"]

    subset = df.head(n).copy()

    if title:
        print(f"\n=== {title} (top {len(subset)}) ===")

    for i, (_, row) in enumerate(subset.iterrows(), 1):
        arch_raw = _safe_get(row, "model.model_architecture", "")
        loss_raw = _safe_get(row, "model.loss", "")
        arch = architecture_mapping.get(arch_raw, arch_raw)
        loss = loss_mapping.get(loss_raw, loss_raw)

        parts = []
        # always try to include these if available
        parts.append(f"Arch={arch}" if arch else None)
        parts.append(f"Loss={loss}" if loss else None)
        parts.append(f"pos_weight={_safe_get(row, 'model.pos_weight')}")
        parts.append(f"alpha={_safe_get(row, 'model.alpha')}")
        parts.append(f"lr={_safe_get(row, 'model.lr')}")
        parts.append(f"batch={_safe_get(row, 'dataloader.batch_size')}")
        parts.append(f"epochs={_safe_get(row, 'training.max_epochs')}")

        # IDs/context
        parts.append(f"join_key={_safe_get(row, 'join_key')}")
        parts.append(f"exp={_safe_get(row, 'experiment')}")
        parts.append(f"date={_safe_get(row, 'experiment_date')}")

        # threshold-table extras if present
        if "threshold" in cols:
            parts.append(f"threshold={_safe_get(row, 'threshold')}")
        if "config_path" in cols:
            parts.append(f"config={_safe_get(row, 'config_path')}")
        if "metrics_path" in cols:
            parts.append(f"metrics={_safe_get(row, 'metrics_path')}")

        # tidy up: drop Nones/empties
        parts = [p for p in parts if p is not None and p.split("=")[-1] not in ("None", "", "nan")]

        print(f"{i:>2}. " + " | ".join(parts))

# --- Use it like this ---
summarize_configs(best_easy.head(), title="Best Easy (by IoU)")
summarize_configs(best_hard.head(), title="Best Hard (by IoU)")


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

# Assumes:
#   arches: list of architectures (e.g., ["deeplabv3_resnet50", "fcn_resnet50", "segformer"])
#   color_by_arch: dict arch -> color
#   map_hard / map_easy: index=arch, column 'join_key'
#   df_thresholds: columns [join_key, threshold, FPR, TP, FP, FN]

# ----------------------------
# Downsampling (choose one)
# ----------------------------
def downsample_stride(df, every=3):
    df = df.sort_values("threshold")
    if every <= 1 or len(df) <= every: return df
    return df.iloc[::every, :]

def downsample_quantiles(df, bins=25):
    df = df.sort_values("threshold")
    if bins <= 1 or len(df) <= bins: return df
    q = np.linspace(0, 1, bins)
    thr = np.quantile(df["threshold"].to_numpy(), q)
    idx = np.abs(df["threshold"].to_numpy()[:, None] - thr[None, :]).argmin(axis=0)
    return df.iloc[sorted(set(idx))]

def downsample_tolerance(df, tol_x=0.002, tol_y=1e-5):
    df = df.sort_values("threshold")
    kept = []
    last_x = last_y = None
    for _, r in df.iterrows():
        x, y = r["threshold"], r["FPR"]
        if last_x is None or abs(x-last_x) >= tol_x or abs(y-last_y) >= tol_y:
            kept.append(r); last_x, last_y = x, y
    return df.iloc[[]] if not kept else type(df)(kept)

# Pick method & params
METHOD, EVERY, BINS, TOL_X, TOL_Y = "stride", 3, 25, 0.002, 1e-5
def apply_downsample(df):
    if METHOD == "quantile":  return downsample_quantiles(df, BINS)
    if METHOD == "tolerance": return downsample_tolerance(df, TOL_X, TOL_Y)
    return downsample_stride(df, EVERY)

# ----------------------------
# Legend order helper (row-major → column-major)
# ----------------------------
def rowmajor_to_colmajor(handles, labels, ncol):
    m = len(handles)
    nrow = int(np.ceil(m / ncol))
    H = [None]*m; L = [None]*m; k = 0
    for i in range(nrow):            # row
        for j in range(ncol):        # col
            idx = j*nrow + i         # column-major index
            if idx < m:
                H[idx] = handles[k]; L[idx] = labels[k]; k += 1
    return H, L

# ----------------------------
# Plot (lines + markers, staggered)
# ----------------------------
OFFSET = 0.001  # stagger x: hard left, easy right
fig, ax = plt.subplots(figsize=(9, 6))

for arch in arches:
    col = color_by_arch[arch]

    # HARD: solid line + circle markers
    if arch in map_hard.index:
        jk_h = map_hard.loc[arch, "join_key"]
        sub_h = df_thresholds[df_thresholds["join_key"] == jk_h]
        if not sub_h.empty:
            sub_h = apply_downsample(sub_h).sort_values("threshold")
            xh = sub_h["threshold"].to_numpy() - OFFSET
            yh = sub_h["FPR"].to_numpy()
            ax.plot(xh, yh, linestyle="-", marker="o", markersize=4,
                    linewidth=1.8, color=col, alpha=0.95)

    # EASY: dashed line + triangle markers
    if arch in map_easy.index:
        jk_e = map_easy.loc[arch, "join_key"]
        sub_e = df_thresholds[df_thresholds["join_key"] == jk_e]
        if not sub_e.empty:
            sub_e = apply_downsample(sub_e).sort_values("threshold")
            xe = sub_e["threshold"].to_numpy() + OFFSET
            ye = sub_e["FPR"].to_numpy()
            ax.plot(xe, ye, linestyle="--", marker="^", markersize=5,
                    linewidth=1.8, color=col, alpha=0.95)

ax.set_xlabel("Threshold")
ax.set_ylabel("False Positive Rate (FPR)")
ax.set_title("FPR vs Threshold")
ax.grid(True, alpha=0.3)

# ----------------------------
# Single legend (row-wise): color=arch, shape/linestyle=set
# ----------------------------
handles_row, labels_row = [], []
for arch in arches:
    col = color_by_arch[arch]
    # Hard: solid + circle
    handles_row.append(Line2D([0],[0], color=col, linestyle="-",
                              marker="o", markersize=6, linewidth=2))
    labels_row.append(f"{arch} (hard)")
    # Easy: dashed + triangle
    handles_row.append(Line2D([0],[0], color=col, linestyle="--",
                              marker="^", markersize=7, linewidth=2))
    labels_row.append(f"{arch} (easy)")

ncol = 2   # e.g., 3 rows x 2 cols if you have 3 architectures
handles_cm, labels_cm = rowmajor_to_colmajor(handles_row, labels_row, ncol)

# --- Your mappings ---
metric_mapping = {
    "metrics.precision": "Precision",
    "metrics.recall": "Recall",
    "metrics.iou": "IoU",
    "metrics.f1score": "F1 Score",
}
architecture_mapping = {
    "deeplabv3_resnet50": "DeepLabV3 (ResNet-50)",
    "fcn_resnet50": "FCN (ResNet-50)",
    "segformer": "SegFormer",
}

# --- Small helpers ---
def pretty_arch(name: str) -> str:
    return architecture_mapping.get(name, name)

def pretty_metric(key: str) -> str:
    # handle common variants (you can extend as needed)
    if key in metric_mapping:
        return metric_mapping[key]
    if key.startswith("metrics.") and key.replace("metrics.", "") in metric_mapping:
        return metric_mapping["metrics." + key.split(".", 1)[1]]
    # common short keys
    alias = {
        "FPR": "FPR",
        "TPR": "TPR",
        "precision": "Precision",
        "recall": "Recall",
        "iou": "IoU",
        "f1score": "F1 Score",
    }
    return alias.get(key, key)

# EXAMPLE: if your Y metric is FPR, make the axis label neat via pretty_metric
Y_METRIC = "FPR"  # change to "precision", "recall", "iou", "f1score", etc.
ax.set_ylabel(pretty_metric(Y_METRIC))

# When you build the single legend, use pretty_arch(...) for labels:
handles_row, labels_row = [], []
for arch in arches:
    col = color_by_arch[arch]
    arch_disp = pretty_arch(arch)

    # Hard (solid+circle)
    handles_row.append(Line2D([0],[0], color=col, linestyle="-",
                              marker="o", markersize=6, linewidth=2))
    labels_row.append(f"{arch_disp} (Small)")

    # Easy (dashed+triangle)
    handles_row.append(Line2D([0],[0], color=col, linestyle="--",
                              marker="^", markersize=7, linewidth=2))
    labels_row.append(f"{arch_disp} (Large)")

# Keep your row-major legend trick:
ncol = 2
handles_cm, labels_cm = rowmajor_to_colmajor(handles_row, labels_row, ncol)
ax.legend(handles=handles_cm, labels=labels_cm,
          title=f"Model (color) • Set (line+marker)",
          ncol=ncol, loc="upper right",
          frameon=True, borderpad=0.6, columnspacing=1.2,
          handletextpad=0.6, markerscale=1.0)

plt.tight_layout()
plt.savefig(f"fpr.png", dpi=1200, bbox_inches="tight", pad_inches=0.02)
plt.show()


# %%
import numpy as np
import pandas as pd

# --- Inputs assumed present:
# df_all: per-run summary table (has 'join_key', 'model.model_architecture', IoU columns, etc.)
# df_thresholds: per-threshold table (has 'join_key', 'TP','FP','TN','FN', and/or 'TPR','FPR')

# ---------------------------------------------------------
# Pick best IoU runs per architecture for Hard/Easy (yours)
# ---------------------------------------------------------
idx_hard = df_all.groupby("model.model_architecture")["metrics.iou_hard"].idxmax()
idx_easy = df_all.groupby("model.model_architecture")["metrics.iou_easy"].idxmax()

best_hard = df_all.loc[idx_hard].assign(Difficulty="Hard")
best_easy = df_all.loc[idx_easy].assign(Difficulty="Easy")

selected = pd.concat([best_hard, best_easy], ignore_index=True)

# ---------------------------------------------------------
# Helpers to build ROC and compute AUC / partial AUC
# ---------------------------------------------------------
def ensure_tpr_fpr(sub: pd.DataFrame) -> pd.DataFrame:
    """Guarantee TPR/FPR columns exist; compute from counts if needed."""
    sub = sub.copy()
    if "TPR" not in sub.columns and {"TP","FN"}.issubset(sub.columns):
        den = (sub["TP"] + sub["FN"]).to_numpy()
        sub["TPR"] = np.where(den > 0, sub["TP"] / den, np.nan)
    if "FPR" not in sub.columns and {"FP","TN"}.issubset(sub.columns):
        den = (sub["FP"] + sub["TN"]).to_numpy()
        sub["FPR"] = np.where(den > 0, sub["FP"] / den, np.nan)
    return sub

def prepare_roc_points(thr_df: pd.DataFrame):
    """Return fpr, tpr sorted by FPR with endpoints padded and values clamped to [0,1]."""
    sub = thr_df[["FPR","TPR"]].dropna().copy()
    if sub.empty:
        return np.array([]), np.array([])
    # sort by FPR; keep the upper envelope at ties
    sub = sub.sort_values(["FPR","TPR"]).groupby("FPR", as_index=False).max()
    fpr = sub["FPR"].to_numpy()
    tpr = sub["TPR"].to_numpy()
    # pad endpoints and clamp
    fpr = np.r_[0.0, fpr, 1.0]
    tpr = np.r_[0.0, tpr, 1.0]
    return np.clip(fpr, 0, 1), np.clip(tpr, 0, 1)

def auc_trapz(fpr, tpr):
    return np.nan if fpr.size < 2 else float(np.trapz(tpr, fpr))

def partial_auc(fpr, tpr, xmax=0.1):
    """Normalized pAUC over [0, xmax]. Set xmax=None to skip."""
    if xmax is None or xmax <= 0 or fpr.size < 2:
        return np.nan
    xmax = min(xmax, 1.0)
    # interpolate TPR at xmax then integrate
    if xmax not in fpr:
        tpr_xmax = np.interp(xmax, fpr, tpr)
        mask = fpr < xmax
        fpr2 = np.r_[fpr[mask], xmax]
        tpr2 = np.r_[tpr[mask], tpr_xmax]
    else:
        idx = np.where(fpr <= xmax)[0]
        fpr2, tpr2 = fpr[idx], tpr[idx]
    # normalize by xmax so pAUC in [0,1]
    return float(np.trapz(tpr2, fpr2) / xmax)

# ---------------------------------------------------------
# Compute AUCs for selected runs
# ---------------------------------------------------------
rows = []
for _, r in selected.iterrows():
    arch = r["model.model_architecture"]
    diff = r["Difficulty"]
    jk = r["join_key"]

    sub = df_thresholds[df_thresholds["join_key"] == jk]
    if sub.empty:
        rows.append({
            "model.model_architecture": arch,
            "Difficulty": diff,
            "join_key": jk,
            "AUC": np.nan,
            "pAUC@0.10": np.nan
        })
        continue

    sub = ensure_tpr_fpr(sub)
    fpr, tpr = prepare_roc_points(sub)
    auc = auc_trapz(fpr, tpr)
    pauc = partial_auc(fpr, tpr, xmax=0.10)  # change cutoff if you prefer

    rows.append({
        "model.model_architecture": arch,
        "Difficulty": diff,
        "join_key": jk,
        "AUC": auc,
        "pAUC@0.10": pauc
    })

auc_df = pd.DataFrame(rows).sort_values(["Difficulty","AUC"], ascending=[True, False]).reset_index(drop=True)

# ---------------------------------------------------------
# Max per split (Hard/Easy) and overall
# ---------------------------------------------------------
max_by_diff = (
    auc_df.loc[auc_df.groupby("Difficulty")["AUC"].idxmax()]
    .reset_index(drop=True)
)
max_overall = auc_df.loc[auc_df["AUC"].idxmax()]

print("Per-model AUCs for best IoU runs:")
print(auc_df)

print("\nMax AUC per split:")
print(max_by_diff[["Difficulty","model.model_architecture","AUC","pAUC@0.10","join_key"]])

print("\nMax AUC overall:")
print(max_overall[["Difficulty","model.model_architecture","AUC","pAUC@0.10","join_key"]])


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

# Assumes:
#   arches: list of architectures (e.g., ["deeplabv3_resnet50", "fcn_resnet50", "segformer"])
#   color_by_arch: dict arch -> color
#   map_hard / map_easy: index=arch, column 'join_key'
#   df_thresholds: columns [join_key, threshold, FPR, TP, FP, FN, ...] and (optionally) TPR

# --- NEW: ensure TPR exists (TPR = TP / (TP + FN)) ---
df_thresholds = df_thresholds.copy()
if "TPR" not in df_thresholds.columns:
    denom = (df_thresholds["TP"] + df_thresholds["FN"]).to_numpy()
    df_thresholds["TPR"] = np.where(denom > 0, df_thresholds["TP"] / denom, np.nan)

# ----------------------------
# Downsampling (choose one)
# ----------------------------
def downsample_stride(df, every=3):
    df = df.sort_values("threshold")
    if every <= 1 or len(df) <= every: return df
    return df.iloc[::every, :]

def downsample_quantiles(df, bins=25):
    df = df.sort_values("threshold")
    if bins <= 1 or len(df) <= bins: return df
    q = np.linspace(0, 1, bins)
    thr = np.quantile(df["threshold"].to_numpy(), q)
    idx = np.abs(df["threshold"].to_numpy()[:, None] - thr[None, :]).argmin(axis=0)
    return df.iloc[sorted(set(idx))]

# --- CHANGED: make tolerance downsampler use a configurable y column (default TPR) ---
def downsample_tolerance(df, tol_x=0.002, tol_y=1e-5, ycol="TPR"):
    df = df.sort_values("threshold")
    kept = []
    last_x = last_y = None
    for _, r in df.iterrows():
        x, y = r["threshold"], r[ycol]
        if last_x is None or abs(x-last_x) >= tol_x or abs(y-last_y) >= tol_y:
            kept.append(r); last_x, last_y = x, y
    return df.iloc[[]] if not kept else type(df)(kept)

# Pick method & params
METHOD, EVERY, BINS, TOL_X, TOL_Y = "stride", 3, 25, 0.002, 1e-5
def apply_downsample(df):
    if METHOD == "quantile":  return downsample_quantiles(df, BINS)
    if METHOD == "tolerance": return downsample_tolerance(df, TOL_X, TOL_Y, ycol="TPR")  # CHANGED
    return downsample_stride(df, EVERY)

# ----------------------------
# Legend order helper (row-major → column-major)
# ----------------------------
def rowmajor_to_colmajor(handles, labels, ncol):
    m = len(handles)
    nrow = int(np.ceil(m / ncol))
    H = [None]*m; L = [None]*m; k = 0
    for i in range(nrow):            # row
        for j in range(ncol):        # col
            idx = j*nrow + i         # column-major index
            if idx < m:
                H[idx] = handles[k]; L[idx] = labels[k]; k += 1
    return H, L

# ----------------------------
# Plot (lines + markers, staggered)
# ----------------------------
OFFSET = 0.001  # stagger x: hard left, easy right
fig, ax = plt.subplots(figsize=(9, 6))

for arch in arches:
    col = color_by_arch[arch]

    # HARD: solid line + circle markers
    if arch in map_hard.index:
        jk_h = map_hard.loc[arch, "join_key"]
        sub_h = df_thresholds[df_thresholds["join_key"] == jk_h]
        if not sub_h.empty:
            sub_h = apply_downsample(sub_h).sort_values("threshold")
            xh = sub_h["threshold"].to_numpy() - OFFSET
            yh = sub_h["TPR"].to_numpy()  # CHANGED from FPR -> TPR
            ax.plot(xh, yh, linestyle="-", marker="o", markersize=4, markevery=5,
                    linewidth=1.8, color=col, alpha=0.95)

    # EASY: dashed line + triangle markers
    if arch in map_easy.index:
        jk_e = map_easy.loc[arch, "join_key"]
        sub_e = df_thresholds[df_thresholds["join_key"] == jk_e]
        if not sub_e.empty:
            sub_e = apply_downsample(sub_e).sort_values("threshold")
            xe = sub_e["threshold"].to_numpy() + OFFSET
            ye = sub_e["TPR"].to_numpy()  # CHANGED from FPR -> TPR
            ax.plot(xe, ye, linestyle="--", marker="^", markersize=5, markevery=5,
                    linewidth=1.8, color=col, alpha=0.95)

ax.set_xlabel("Threshold")
# (we'll set ylabel via pretty_metric below)
ax.set_title("TPR vs Threshold")  # CHANGED
ax.grid(True, alpha=0.3)

# ----------------------------
# Single legend (row-wise): color=arch, shape/linestyle=set
# ----------------------------
handles_row, labels_row = [], []
for arch in arches:
    col = color_by_arch[arch]
    # Hard: solid + circle
    handles_row.append(Line2D([0],[0], color=col, linestyle="-",
                              marker="o", markersize=6, linewidth=2))
    labels_row.append(f"{arch} (hard)")
    # Easy: dashed + triangle
    handles_row.append(Line2D([0],[0], color=col, linestyle="--",
                              marker="^", markersize=7, linewidth=2))
    labels_row.append(f"{arch} (easy)")

ncol = 2
handles_cm, labels_cm = rowmajor_to_colmajor(handles_row, labels_row, ncol)

# --- Your mappings ---
metric_mapping = {
    "metrics.precision": "Precision",
    "metrics.recall": "Recall",
    "metrics.iou": "IoU",
    "metrics.f1score": "F1 Score",
}
architecture_mapping = {
    "deeplabv3_resnet50": "DeepLabV3 (ResNet-50)",
    "fcn_resnet50": "FCN (ResNet-50)",
    "segformer": "SegFormer",
}

# --- Small helpers ---
def pretty_arch(name: str) -> str:
    return architecture_mapping.get(name, name)

def pretty_metric(key: str) -> str:
    if key in metric_mapping:
        return metric_mapping[key]
    if key.startswith("metrics.") and key.replace("metrics.", "") in metric_mapping:
        return metric_mapping["metrics." + key.split(".", 1)[1]]
    alias = {
        "FPR": "FPR",
        "TPR": "TPR",
        "precision": "Precision",
        "recall": "Recall",
        "iou": "IoU",
        "f1score": "F1 Score",
    }
    return alias.get(key, key)

# --- CHANGED: Y metric is now TPR ---
Y_METRIC = "TPR"
ax.set_ylabel(pretty_metric(Y_METRIC))

# Optional: nicer legend labels using pretty_arch(...)
handles_row, labels_row = [], []
for arch in arches:
    col = color_by_arch[arch]
    arch_disp = pretty_arch(arch)
    handles_row.append(Line2D([0],[0], color=col, linestyle="-",
                              marker="o", markersize=6, linewidth=2))
    labels_row.append(f"{arch_disp} (Small)")   # keep your naming if desired
    handles_row.append(Line2D([0],[0], color=col, linestyle="--",
                              marker="^", markersize=7, linewidth=2))
    labels_row.append(f"{arch_disp} (Large)")

handles_cm, labels_cm = rowmajor_to_colmajor(handles_row, labels_row, ncol)
ax.legend(handles=handles_cm, labels=labels_cm,
          title=f"Model (color) • Set (line+marker)",
          ncol=ncol, loc="upper right",
          frameon=True, borderpad=0.6, columnspacing=1.2,
          handletextpad=0.6, markerscale=1.0)

plt.tight_layout()
plt.savefig(f"tpr.png", dpi=1200, bbox_inches="tight", pad_inches=0.02)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Assumes you already have:
# - arches: list[str]
# - color_by_arch: dict[str, color]
# - map_hard / map_easy: DataFrames indexed by arch with column 'join_key'
# - df_thresholds with columns: [join_key, threshold, TN, FP, FN, TP, TPR, FPR, ...]

# --- Ensure TPR/FPR exist from counts if needed ---
df_thresholds = df_thresholds.copy()
if "TPR" not in df_thresholds.columns and {"TP","FN"}.issubset(df_thresholds.columns):
    den = (df_thresholds["TP"] + df_thresholds["FN"]).to_numpy()
    df_thresholds["TPR"] = np.where(den > 0, df_thresholds["TP"] / den, np.nan)
if "FPR" not in df_thresholds.columns and {"FP","TN"}.issubset(df_thresholds.columns):
    den = (df_thresholds["FP"] + df_thresholds["TN"]).to_numpy()
    df_thresholds["FPR"] = np.where(den > 0, df_thresholds["FP"] / den, np.nan)

# --- Helpers ---
def _roc_prepare(sub):
    """Return FPR, TPR sorted by FPR, with (0,0) and (1,1) padded if missing."""
    sub = sub[["FPR","TPR"]].dropna().copy()
    if sub.empty:
        return np.array([]), np.array([])
    # sort by FPR; if ties, keep max TPR so curve is non-decreasing in y
    sub = sub.sort_values(["FPR","TPR"]).groupby("FPR", as_index=False).max()
    fpr = sub["FPR"].to_numpy()
    tpr = sub["TPR"].to_numpy()
    # pad endpoints
    if fpr.size and (fpr[0] > 0 or tpr[0] > 0):
        fpr = np.r_[0.0, fpr]; tpr = np.r_[0.0, tpr]
    if fpr.size and (fpr[-1] < 1 or tpr[-1] < 1):
        fpr = np.r_[fpr, 1.0]; tpr = np.r_[tpr, 1.0]
    # clamp
    fpr = np.clip(fpr, 0, 1); tpr = np.clip(tpr, 0, 1)
    return fpr, tpr

def _auc_trapz(fpr, tpr):
    """Trapezoidal AUC; returns np.nan if insufficient points."""
    if fpr.size < 2: return np.nan
    return float(np.trapz(tpr, fpr))

# Optional downsample to lighten curves (tolerance in x,y space)
def downsample_tolerance_xy(x, y, tol_x=0.0015, tol_y=0.0015):
    if x.size <= 2: return x, y
    keep_x, keep_y = [x[0]], [y[0]]
    last_x, last_y = x[0], y[0]
    for xi, yi in zip(x[1:], y[1:]):
        if abs(xi - last_x) >= tol_x or abs(yi - last_y) >= tol_y:
            keep_x.append(xi); keep_y.append(yi)
            last_x, last_y = xi, yi
    if keep_x[-1] != x[-1]:
        keep_x.append(x[-1]); keep_y.append(y[-1])
    return np.asarray(keep_x), np.asarray(keep_y)

# --- Plot ---
fig, ax = plt.subplots(figsize=(8.5, 6))
handles_row, labels_row = [], []

for arch in arches:
    col = color_by_arch[arch]

    # HARD (solid)
    if arch in map_hard.index:
        jk = map_hard.loc[arch, "join_key"]
        sub = df_thresholds[df_thresholds["join_key"] == jk]
        fpr, tpr = _roc_prepare(sub)
        if fpr.size:
            fpr_d, tpr_d = downsample_tolerance_xy(fpr, tpr)
            auc = _auc_trapz(fpr, tpr)
            ax.plot(fpr_d, tpr_d, linestyle="-", marker=None, linewidth=2.0, alpha=0.95, color=col)
            handles_row.append(Line2D([0],[0], color=col, linestyle="-", linewidth=2.5))
            labels_row.append(f"{arch} (Hard) — AUC={auc:.3f}")

    # EASY (dashed)
    if arch in map_easy.index:
        jk = map_easy.loc[arch, "join_key"]
        sub = df_thresholds[df_thresholds["join_key"] == jk]
        fpr, tpr = _roc_prepare(sub)
        if fpr.size:
            fpr_d, tpr_d = downsample_tolerance_xy(fpr, tpr)
            auc = _auc_trapz(fpr, tpr)
            ax.plot(fpr_d, tpr_d, linestyle="--", marker=None, linewidth=2.0, alpha=0.95, color=col)
            handles_row.append(Line2D([0],[0], color=col, linestyle="--", linewidth=2.5))
            labels_row.append(f"{arch} (Easy) — AUC={auc:.3f}")

# Diagonal chance line
ax.plot([0,1], [0,1], linestyle=":", linewidth=1.5, color="gray", alpha=0.8)

ax.set_xlabel("False Positive Rate (FPR)")
ax.set_ylabel("True Positive Rate (TPR)")
ax.set_title("ROC Curve — TPR vs FPR")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# Legend
ax.legend(handles_row, labels_row, loc="lower right", frameon=True, borderpad=0.6)

plt.tight_layout()
plt.savefig(f"roc.png", dpi=1200, bbox_inches="tight", pad_inches=0.02)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- Config knobs ----
VIEW = "logx"          # "linear" | "logx" | "logit" | "det" | "inset"
XMIN, XMAX = 1e-5, 0.1 # x-window (FPR) to zoom; set None to skip
SHOW_PARC = True       # show partial AUC up to XMAX (if set)
EPS = 1e-6             # to avoid 0/1 issues on log/logit/probit scales

# ---- Assumes arches, color_by_arch, map_hard, map_easy, df_thresholds with FPR & TPR ----
df_thresholds = df_thresholds.copy()
if "TPR" not in df_thresholds.columns and {"TP","FN"}.issubset(df_thresholds.columns):
    den = (df_thresholds["TP"] + df_thresholds["FN"]).to_numpy()
    df_thresholds["TPR"] = np.where(den > 0, df_thresholds["TP"] / den, np.nan)
if "FPR" not in df_thresholds.columns and {"FP","TN"}.issubset(df_thresholds.columns):
    den = (df_thresholds["FP"] + df_thresholds["TN"]).to_numpy()
    df_thresholds["FPR"] = np.where(den > 0, df_thresholds["FP"] / den, np.nan)

def _roc_prepare(sub):
    sub = sub[["FPR","TPR"]].dropna().copy()
    if sub.empty: return np.array([]), np.array([])
    sub = sub.sort_values(["FPR","TPR"]).groupby("FPR", as_index=False).max()
    fpr, tpr = sub["FPR"].to_numpy(), sub["TPR"].to_numpy()
    # pad and clamp
    fpr = np.r_[0.0, fpr, 1.0]; tpr = np.r_[0.0, tpr, 1.0]
    fpr = np.clip(fpr, 0, 1); tpr = np.clip(tpr, 0, 1)
    return fpr, tpr

def _auc_trapz(fpr, tpr): 
    return np.nan if fpr.size < 2 else float(np.trapz(tpr, fpr))

def _partial_auc(fpr, tpr, xmax):
    if xmax is None or fpr.size < 2: return np.nan
    xmax = min(max(xmax, 0.0), 1.0)
    # interpolate TPR at fpr=xmax then integrate on [0, xmax]
    if xmax not in fpr:
        tpr_xmax = np.interp(xmax, fpr, tpr)
        fpr2 = np.r_[fpr[fpr < xmax], xmax]
        tpr2 = np.r_[tpr[fpr < xmax], tpr_xmax]
    else:
        idx = np.where(fpr <= xmax)[0]
        fpr2, tpr2 = fpr[idx], tpr[idx]
    return float(np.trapz(tpr2, fpr2) / xmax) if xmax > 0 else np.nan  # normalized pAUC

def _downsample_xy(x, y, tol_x=0.0015, tol_y=0.0015):
    if x.size <= 2: return x, y
    keep_x, keep_y = [x[0]], [y[0]]
    last_x, last_y = x[0], y[0]
    for xi, yi in zip(x[1:], y[1:]):
        if abs(xi - last_x) >= tol_x or abs(yi - last_y) >= tol_y:
            keep_x.append(xi); keep_y.append(yi)
            last_x, last_y = xi, yi
    if keep_x[-1] != x[-1]:
        keep_x.append(x[-1]); keep_y.append(y[-1])
    return np.asarray(keep_x), np.asarray(keep_y)

# ---- Optional transforms for scale ----
def _probit(p):  # DET-style axis
    # clamp away from 0/1 to avoid inf
    p = np.clip(p, EPS, 1-EPS)
    # Φ^{-1}(p) via inverse error function if SciPy not available
    try:
        from scipy.stats import norm
        return norm.ppf(p)
    except Exception:
        # approximate inverse-CDF using erfinv if available
        try:
            from math import sqrt
            import numpy as _np
            from numpy import erfinv
            return sqrt(2) * erfinv(2*p - 1)
        except Exception:
            # fallback: logit as a rough alternative
            return np.log(p/(1-p))

fig, ax = plt.subplots(figsize=(8.8, 6.2))
handles, labels = [], []

for arch in arches:
    col = color_by_arch[arch]

    for split_name, mapping, style in [("Hard", map_hard, "-"), ("Easy", map_easy, "--")]:
        if arch not in mapping.index: continue
        jk = mapping.loc[arch, "join_key"]
        fpr, tpr = _roc_prepare(df_thresholds[df_thresholds["join_key"] == jk])
        if not fpr.size: continue

        # downsample for visual clarity (on linear coords)
        fpr_d, tpr_d = _downsample_xy(fpr, tpr)

        label = f"{arch} ({split_name}) — AUC={_auc_trapz(fpr, tpr):.3f}"
        if SHOW_PARC and XMAX is not None:
            parc = _partial_auc(fpr, tpr, XMAX)
            if not np.isnan(parc):
                label += f", pAUC[0,{XMAX:g}]={parc:.3f}"

        # apply scaling/transform
        if VIEW == "det":
            xplot, yplot = _probit(fpr_d), _probit(1 - tpr_d)  # x=FPR, y=FNR on probit scale
        else:
            xplot, yplot = fpr_d, tpr_d

        ax.plot(xplot, yplot, linestyle=style, linewidth=2.0, color=col, alpha=0.97, label=label)

# Baselines
if VIEW == "det":
    ax.set_xlabel("FPR (probit)")
    ax.set_ylabel("FNR (probit)")
    ax.set_title("DET Curve (probit scale)")
else:
    # diagonal chance line in ROC space
    ax.plot([0,1], [0,1], linestyle=":", linewidth=1.4, color="gray", alpha=0.8)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ttl = "ROC (TPR vs FPR)"
    if VIEW == "logx": ttl += " — log-scaled FPR"
    if VIEW == "logit": ttl += " — logit-scaled FPR"
    ax.set_title(ttl)

    # axis scaling
    if VIEW == "logx":
        # avoid zeros on log scale
        ax.set_xscale("log")
        ax.set_xlim(left=EPS, right=1.0)
    elif VIEW == "logit":
        # probabilities in (0,1); push away from edges
        ax.set_xscale("logit")
        ax.set_xlim(EPS, 1 - EPS)

# Region zoom (linear/log/logit all support limits)
if VIEW != "det" and (XMIN is not None or XMAX is not None):
    lo = EPS if XMIN is None else max(XMIN, EPS)
    hi = 1.0 if XMAX is None else min(XMAX, 1.0)
    ax.set_xlim(lo, hi)

ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", frameon=True, borderpad=0.6)
plt.tight_layout()

# Optional inset for ultra-low FPR view
if VIEW == "inset":
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    axins = inset_axes(ax, width="50%", height="50%", loc="lower right", borderpad=1.2)
    axins.set_xlim(EPS, XMAX or 0.01)
    axins.set_ylim(0.9, 1.001)
    axins.set_xscale("log")
    # replot the same lines into inset (reuse legend labels/colors)
    for line in ax.get_lines()[:-1]:  # skip the chance diagonal
        x, y = line.get_xdata(), line.get_ydata()
        axins.plot(x, y, linestyle=line.get_linestyle(), color=line.get_color(), linewidth=1.6, alpha=0.95)
    axins.grid(True, alpha=0.3)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.6")
plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------
# Load & prep
# ----------------------------
df_all = pd.read_csv("data/all_results_v2.csv")
df_thresholds = pd.read_csv("data/thresholded_results.csv")

# Pick best (max IoU) run per architecture for Hard/Easy
idx_hard = df_all.groupby("model.model_architecture")["metrics.iou_hard"].idxmax()
idx_easy = df_all.groupby("model.model_architecture")["metrics.iou_easy"].idxmax()

best_hard = df_all.loc[idx_hard].assign(Difficulty="Hard")
best_easy = df_all.loc[idx_easy].assign(Difficulty="Easy")

map_hard = best_hard.set_index("model.model_architecture")[["join_key"]]
map_easy = best_easy.set_index("model.model_architecture")[["join_key"]]

arches = sorted(df_all["model.model_architecture"].unique())
default_colors = plt.cm.tab10.colors
color_by_arch = {arch: default_colors[i % len(default_colors)] for i, arch in enumerate(arches)}

architecture_mapping = {
    "deeplabv3_resnet50": "DeepLabV3 (ResNet-50)",
    "fcn_resnet50": "FCN (ResNet-50)",
    "segformer": "SegFormer",
}
def pretty_arch(name: str) -> str:
    return architecture_mapping.get(name, name)

# ----------------------------
# Downsampling (optional)
# ----------------------------
def downsample_stride_xy(x, y, every=3):
    if every <= 1 or len(x) <= every: return x, y
    return x[::every], y[::every]

METHOD_EVERY = 2     # 1 = no downsample; try 2–4 for dense curves

# ----------------------------
# PR curve utilities
# ----------------------------
def compute_pr_from_counts(df_thr):
    """
    Expects columns: TP, FP, FN (per-threshold rows).
    Returns recall (asc), precision aligned to recall.
    """
    TP = df_thr["TP"].astype(float).to_numpy()
    FP = df_thr["FP"].astype(float).to_numpy()
    FN = df_thr["FN"].astype(float).to_numpy()

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(TP + FP > 0, TP / (TP + FP), 1.0)  # convention: precision=1 when no preds
        recall    = np.where(TP + FN > 0, TP / (TP + FN), 0.0)

    # Sort by recall (x-axis)
    idx = np.argsort(recall)
    r = recall[idx]
    p = precision[idx]

    # Add endpoints to make a complete PR curve:
    # start at recall=0 with max precision at lowest recall; end at recall=1 with precision=0 (if needed)
    if r[0] > 0:
        r = np.concatenate(([0.0], r))
        p = np.concatenate(([p[0]], p))
    if r[-1] < 1:
        r = np.concatenate((r, [1.0]))
        p = np.concatenate((p, [0.0]))

    return r, p

def auc_pr_trapz(r, p):
    # Ensure monotonic increasing recall for integration
    order = np.argsort(r)
    r2, p2 = r[order], p[order]
    return float(np.trapz(p2, r2))

def auc_pr_ap(r, p):
    """
    Interpolated Average Precision:
    - make precision a non-increasing function of recall (right-side envelope)
    - integrate step-wise precision over recall changes
    """
    order = np.argsort(r)
    r, p = r[order], p[order]
    # Right-side precision envelope (monotone non-increasing)
    p = np.maximum.accumulate(p[::-1])[::-1]
    # Step integral: sum precision * delta_recall
    dr = np.diff(r)
    return float(np.sum(p[:-1] * dr))

AUC_METHOD = "ap"   # "ap" for Average Precision; "trapz" for straight trapezoid

def auc_pr(r, p):
    return auc_pr_ap(r, p) if AUC_METHOD == "ap" else auc_pr_trapz(r, p)

# ----------------------------
# Legend layout helper
# ----------------------------
def rowmajor_to_colmajor(handles, labels, ncol):
    m = len(handles)
    nrow = int(np.ceil(m / ncol))
    H = [None]*m; L = [None]*m; k = 0
    for i in range(nrow):            # row
        for j in range(ncol):        # col
            idx = j*nrow + i
            if idx < m:
                H[idx] = handles[k]; L[idx] = labels[k]; k += 1
    return H, L

# ----------------------------
# Plot PR curves
# ----------------------------
fig, ax = plt.subplots(figsize=(9, 6))
OFFSET_R = 0.0   # no need to stagger recall; PR curves share same x

auc_by_series = {}

for arch in arches:
    col = color_by_arch[arch]

    # HARD (shown as "Small" in your legend style): solid + circle
    if arch in map_hard.index:
        jk_h = map_hard.loc[arch, "join_key"]
        sub_h = df_thresholds[df_thresholds["join_key"] == jk_h]
        if not sub_h.empty:
            r, p = compute_pr_from_counts(sub_h)
            # optional downsample for speed/clarity
            r_ds, p_ds = downsample_stride_xy(r, p, every=METHOD_EVERY)
            ax.plot(r_ds, p_ds, linestyle="-", marker="o", markersize=4, markevery=5,
                    linewidth=1.8, color=col, alpha=0.95)
            auc_by_series[(arch, "Small")] = auc_pr(r, p)

    # EASY (shown as "Large"): dashed + triangle
    if arch in map_easy.index:
        jk_e = map_easy.loc[arch, "join_key"]
        sub_e = df_thresholds[df_thresholds["join_key"] == jk_e]
        if not sub_e.empty:
            r, p = compute_pr_from_counts(sub_e)
            r_ds, p_ds = downsample_stride_xy(r, p, every=METHOD_EVERY)
            ax.plot(r_ds, p_ds, linestyle="--", marker="^", markersize=5, markevery=5,
                    linewidth=1.8, color=col, alpha=0.95)
            auc_by_series[(arch, "Large")] = auc_pr(r, p)

# Axes & baseline
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"Precision–Recall Curves (PR-AUC = {AUC_METHOD.upper()})")
ax.grid(True, alpha=0.3)

# Optional random-chance baseline for heavily imbalanced data:
# If you know the positive prior π = P(y=1) for the dataset, precision baseline = π.
# For lack of a global prior here, we skip plotting a single line.

# Legend with AUCs appended
handles_row, labels_row = [], []
for arch in arches:
    col = color_by_arch[arch]
    arch_disp = pretty_arch(arch)

    auc_small = auc_by_series.get((arch, "Small"))
    lab_small = f"{arch_disp} (Small)"
    if auc_small is not None:
        lab_small += f" — AUC={auc_small:.3f}"
    handles_row.append(Line2D([0],[0], color=col, linestyle="-",
                              marker="o", markersize=6, linewidth=2))
    labels_row.append(lab_small)

    auc_large = auc_by_series.get((arch, "Large"))
    lab_large = f"{arch_disp} (Large)"
    if auc_large is not None:
        lab_large += f" — AUC={auc_large:.3f}"
    handles_row.append(Line2D([0],[0], color=col, linestyle="--",
                              marker="^", markersize=7, linewidth=2))
    labels_row.append(lab_large)

ncol = 2
handles_cm, labels_cm = rowmajor_to_colmajor(handles_row, labels_row, ncol)
# keep full PR unit square
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)

# rebuild (or keep) your 2-column legend entries
ncol = 2
handles_cm, labels_cm = rowmajor_to_colmajor(handles_row, labels_row, ncol)

# INSIDE, top-right, compact
leg = ax.legend(
    handles=handles_cm,
    labels=labels_cm,
    title="Model (color) • Plume Size (line+marker)",
    ncol=ncol,
    loc="upper right",          # inside the axes
    bbox_to_anchor=(0.98, 0.98),# slight inset from the corner
    frameon=True,
    fancybox=True,
    framealpha=0.9,
    borderpad=0.3,
    labelspacing=0.25,
    columnspacing=0.6,
    handletextpad=0.4,
    handlelength=1.4,
    markerscale=0.8,
    fontsize="x-small",
    title_fontsize="x-small",
)

# subtle frame styling
leg.get_frame().set_linewidth(0.8)
leg.get_frame().set_edgecolor((0, 0, 0, 0.25))

# no extra top margin needed since legend is inside
plt.tight_layout()
plt.savefig("pr_auc.png", dpi=1200, bbox_inches="tight", pad_inches=0.02)
plt.show()


# %%



