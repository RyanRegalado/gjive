import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from pathlib import Path
from autism.utils import get_region

DIR = Path().cwd() / "autism"

# ============================================================
# Load coefficient matrix
# ============================================================

beta = pd.read_csv(
    DIR / "csvs" / "logistic_beta_matrix.csv",
    index_col=0
)

beta.index = beta.index.astype(int)
beta.columns = beta.columns.astype(int)

# ============================================================
# Load ROI labels
# ============================================================

roi = pd.read_csv(
    DIR / "csvs" / "CC200_ROI_labels.csv"
)

roi["Region"] = roi["AAL"].apply(get_region)

roi = roi.dropna(subset=["Region"])

roi = roi[
    roi["ROI number"].isin(beta.index)
]

# ============================================================
# Sort ROIs by region
# ============================================================

roi = roi.sort_values(
    ["Region", "ROI number"]
).reset_index(drop=True)

ordered_rois = roi["ROI number"].astype(int).tolist()

beta_sorted = beta.loc[
    ordered_rois,
    ordered_rois
]

# ============================================================
# Compute block boundaries
# ============================================================

boundaries = []
centers = []
labels = []

start = 0

for region, group in roi.groupby("Region"):

    size = len(group)

    boundaries.append(start)

    centers.append(start + size/2)

    labels.append(region)

    start += size

boundaries.append(len(ordered_rois))

# ============================================================
# Plot
# ============================================================

# Make the figure slightly taller and wider to give elements breathing room
plt.figure(figsize=(15, 15))

ax = sns.heatmap(
    beta_sorted,
    cmap="RdBu_r",
    center=0,
    square=True,
    xticklabels=False,
    yticklabels=False,
    cbar_kws={
        "label": "Logistic Regression Coefficient",
        "shrink": 0.8,  # Slightly scales down colorbar height to match plot nicely
    },
)

# ------------------------------------------------------------
# Draw anatomical boundaries
# ------------------------------------------------------------

for b in boundaries:
  ax.axhline(b, color="black", linewidth=0.7)
  ax.axvline(b, color="black", linewidth=0.7)

# ------------------------------------------------------------
# Region labels
# ------------------------------------------------------------

ax.set_xticks(centers)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)

ax.set_yticks(centers)
ax.set_yticklabels(labels, fontsize=10)

# Adjust margins so the labels fit nicely without needing 'bbox_inches=tight'
plt.subplots_adjust(bottom=0.20, left=0.22, top=0.92, right=0.90)

plt.savefig(DIR / "grouped_coefficient_heatmap.png", dpi=300)

plt.show()