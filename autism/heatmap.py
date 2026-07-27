# autism/heatmap.py
"""
Heatmap of U0 U0.T - U1 U1.T, with rows/columns reordered by brain region.

Region order: Cerebellum, Frontal, Occipital, Parietal, Temporal
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gjive.estimate import estimate_data
from gjive.estimate_spec import EstimateSpec
from gjive.dataset import GjiveData

from autism.run_gjive import transform, procrustes_transformation
from autism.utils import get_region


DATA_PATH = Path.cwd() / "autism" / "csvs" / "processed_connectivity_dataset.csv"
ROI_PATH = Path.cwd() / "autism" / "csvs" / "CC200_ROI_labels.csv"
FIG_SAVE_PATH = Path.cwd() / "autism" / "figures" / "heatmap"

REGION_ORDER = ["Cerebellum", "Frontal", "Occipital", "Parietal", "Temporal"]


def get_regions_list() -> list[str]:
    aals = pd.read_csv(ROI_PATH)["AAL"].tolist()
    return [get_region(aal) for aal in aals]


def region_keep_mask(regions: list[str]) -> np.ndarray:
    """
    Boolean mask: True for ROIs whose region is in REGION_ORDER,
    False for anything else (e.g. subcortical/limbic/"Other"/None),
    which get excluded from the heatmap entirely.
    """
    regions_arr = np.asarray(regions)
    return np.isin(regions_arr, REGION_ORDER)


def region_sort_order(regions: list[str]) -> np.ndarray:
    """
    Return indices (into the already-filtered region list) that reorder
    `regions` according to REGION_ORDER. Assumes `regions` has already
    been filtered via region_keep_mask -- i.e. every entry is in
    REGION_ORDER.
    """
    regions_arr = np.asarray(regions)

    rank_map = {region: i for i, region in enumerate(REGION_ORDER)}
    rank_keys = np.array([rank_map[r] for r in regions_arr])

    order = np.argsort(rank_keys, kind="stable")
    return order

def region_boundaries(sorted_regions: list[str]) -> list[tuple[str, int, int]]:
    """
    Given regions already sorted per REGION_ORDER, return a list of
    (region_name, start_idx, end_idx) blocks for drawing gridlines/labels.
    """
    boundaries = []
    start = 0
    current = sorted_regions[0]

    for i in range(1, len(sorted_regions) + 1):
        if i == len(sorted_regions) or sorted_regions[i] != current:
            boundaries.append((current, start, i))
            if i < len(sorted_regions):
                current = sorted_regions[i]
                start = i

    return boundaries


def plot_heatmap(
    diff_matrix: np.ndarray,
    sorted_regions: list[str],
    save_path: Path,
    fname: str,
    vmax: float | None = None,
):
    boundaries = region_boundaries(sorted_regions)

    if vmax is None:
        vmax = np.abs(diff_matrix).max()
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(diff_matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax)

    # gridlines + tick labels at region boundaries
    tick_positions = []
    tick_labels = []
    for name, start, end in boundaries:
        mid = (start + end) / 2
        tick_positions.append(mid)
        tick_labels.append(name)

        # draw boundary lines (skip the very first at 0)
        if start != 0:
            ax.axhline(start - 0.5, color="black", linewidth=0.8)
            ax.axvline(start - 0.5, color="black", linewidth=0.8)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    ax.set_title(r"$U_0 U_0^T - U_1 U_1^T$ (regions grouped)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Difference")

    plt.tight_layout()

    save_path.mkdir(parents=True, exist_ok=True)
    out_path = save_path / fname
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Heatmap saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Heatmap of U0 U0.T - U1 U1.T, rows/cols grouped by brain region."
    )
    parser.add_argument("--r", type=int, required=True, help="Joint rank r")
    parser.add_argument("--rfk", type=int, nargs="+", required=True,
                         help="Group-specific ranks, one per group (e.g. --rfk 12 12)")
    parser.add_argument("--rk", type=int, required=True,
                         help="Individual rank rk, applied uniformly to all subjects")
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)

    print("Transforming dataframe...")
    X, group_assignments = transform(df)
    K = X.shape[0]

    data = GjiveData.from_real_data(X, group_assignments - 1)

    est_spec = EstimateSpec(
        r=args.r,
        rfk=args.rfk,
        rk=[args.rk] * K,
    )

    print("Estimating...")
    estimate = estimate_data(data, est_spec)
    print("Estimation success!")

    Uf0 = estimate.get_Uf(0)
    Uf1 = estimate.get_Uf(1)

    r = min(Uf0.shape[1], Uf1.shape[1])
    Uf0 = Uf0[:, :r]
    Uf1 = Uf1[:, :r]

    Uf1_tilde = procrustes_transformation(Uf1, Uf0)

    diff = Uf0 @ Uf0.T - Uf1_tilde @ Uf1_tilde.T

    print("Mapping rows to regions...")
    regions = get_regions_list()

    if len(regions) != diff.shape[0]:
        raise ValueError(
            f"Number of ROI region labels ({len(regions)}) does not match "
            f"matrix dimension ({diff.shape[0]})."
        )

    keep = region_keep_mask(regions)
    n_dropped = (~keep).sum()
    if n_dropped:
        print(f"Excluding {n_dropped} ROI(s) not in REGION_ORDER "
              f"(subcortical/limbic/unlabeled): {[r for r in regions if r not in REGION_ORDER]}")


    diff_kept = diff[np.ix_(keep, keep)]
    regions_kept = [r for r, k in zip(regions, keep) if k]

    order = region_sort_order(regions_kept)
    diff_sorted = diff_kept[np.ix_(order, order)]
    sorted_regions = [regions_kept[i] for i in order]

    rfk_str = "_".join(str(v) for v in args.rfk)
    fname = f"heatmap_ranks_{args.r}_{rfk_str}_{args.rk}.png"

    print("Plotting heatmap...")
    plot_heatmap(diff_sorted, sorted_regions, FIG_SAVE_PATH, fname)


if __name__ == "__main__":
    main()