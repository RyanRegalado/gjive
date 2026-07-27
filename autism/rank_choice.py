# autism/rank_choice.py
"""
Rank estimation workflow: for a given full_rank budget, build ANY valid
(r, rfk, rk) split satisfying:
    r, rfk, rk > 0
    rfk >= 3
    r + rfk + rk == full_rank

Since M_joint's per-subject truncation only depends on the TOTAL
r + rfk[group] + rk[i], not the individual split values, any valid split
produces the identical M_joint matrix. So we only need one split per
full_rank value to compute the spectrum.

Plot the singular values (scree plot) of M_joint and run get_elbows() --
the resulting elbow is the estimate for r.
"""

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from autism.run_gjive import transform
from autism.utils import get_elbows

from gjive import GjiveData, U_joint
from gjive.utils import M_joint, M_group


DATA_PATH = Path.cwd() / "autism" / "csvs" / "processed_connectivity_dataset.csv"
FIG_ROOT = Path.cwd() / "autism" / "figures" / "scree_plot_full_rank"
FIG_ROOT_GROUP = Path.cwd() / "autism" / "figures" / "scree_plot_group"


def plot_group_scree(
    values: np.ndarray,
    r: int,
    rfk: int,
    rk: int,
    group_id: int,
    xlim: int | None = None,
) -> list[int]:
    FIG_ROOT_GROUP.mkdir(parents=True, exist_ok=True)

    elbows = get_elbows(values)

    plt.figure(figsize=(9, 6))
    x = np.arange(1, len(values) + 1)
    plt.plot(x, values, marker="o", markersize=3, linewidth=1, color="darkorange")

    colors = plt.cm.tab10.colors
    for i, e in enumerate(elbows):
        plt.axvline(e, color=colors[i % len(colors)], linestyle="--", alpha=0.7,
                    label=f"elbow {i+1}: {e}")
        if e - 1 < len(values):
            plt.scatter([e], [values[e - 1]], color=colors[i % len(colors)], zorder=5)

    if xlim:
        plt.xlim(0, 25)

    plt.xlabel("Component index")
    plt.ylabel("Singular value")
    plt.title(f"M_group[{group_id}] singular values (r={r}, split used: rfk={rfk}, rk={rk})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    suffix = "" if xlim is None else "_zoom"
    fname = f"m_group{group_id}_r{r}{suffix}.png"
    out_path = FIG_ROOT_GROUP / fname
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"  group {group_id} elbow candidates: {elbows}")
    print(f"  saved: {out_path}")

    return elbows


def run_group_scan(
    data: GjiveData,
    r: int,
    remaining: int,
    group_assignments: Sequence[int],
    n_groups: int,
):
    """
    Second stage: given r (from stage one), build U_hat via U_joint,
    then for each group compute M_group using ANY valid (rfk, rk) split
    summing to `remaining` (r + rfk + rk == full_rank => remaining =
    full_rank - r). Plot the scree and read the elbow as the rfk
    estimate for that group.
    """
    K = data.A.shape[0]

    r_local, rfk, rk = make_valid_split(remaining + 1)  # reuse helper: r>=1,rfk>=3,rk>=1
    # make_valid_split enforces its own r>=1 slot we don't need here --
    # instead build the (rfk, rk) split directly:
    rfk = 3
    rk = remaining - rfk
    if rk <= 0:
        raise ValueError(
            f"remaining={remaining} too small for rfk>=3, rk>=1 "
            f"(need remaining >= 4)."
        )

    print(f"\n===== Group scan: r={r}, remaining={remaining} (split rfk={rfk}, rk={rk}) =====")

    U_hat = U_joint(
        data.A, r,
        [rfk] * n_groups,
        [rk] * K,
        group_assignments,
    )

    results = {}
    for group_id in range(n_groups):
        M = M_group(
            A=data.A,
            U=U_hat,
            rfk=[rfk] * n_groups,
            rk=[rk] * K,
            group_assignments=group_assignments,
            group_id=group_id,
        )

        singular_values = np.linalg.eigvalsh(M)[::-1]

        elbows = plot_group_scree(singular_values, r, rfk, rk, group_id)
        plot_group_scree(singular_values, r, rfk, rk, group_id, xlim=min(50, remaining))

        results[group_id] = {
            "singular_values": singular_values,
            "elbows": elbows,
            "rfk_estimate": elbows[0] if elbows else None,
        }
        print(f"  -> group {group_id} rfk estimate (first elbow): {results[group_id]['rfk_estimate']}")

    return results, U_hat


# ----------------------------------------------------------------------
# Split construction
# ----------------------------------------------------------------------

def make_valid_split(full_rank: int) -> tuple[int, int, int]:
    """
    Build any single valid (r, rfk, rk) split summing to full_rank,
    satisfying r, rfk, rk > 0 and rfk >= 3.

    The specific split doesn't matter for M_joint's spectrum (only the
    total does), so we use a simple minimal-r, minimal-rfk split and
    push the remainder into rk.
    """
    r = 1
    rfk = 3
    rk = full_rank - r - rfk

    if rk <= 0:
        raise ValueError(
            f"full_rank={full_rank} is too small to satisfy "
            f"r>=1, rfk>=3, rk>=1 (minimum full_rank is 5)."
        )

    return r, rfk, rk


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_scree(
    values: np.ndarray,
    full_rank: int,
    r: int,
    rfk: int,
    rk: int,
    xlim: int | None = None,
) -> list[int]:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    elbows = get_elbows(values)

    plt.figure(figsize=(9, 6))
    x = np.arange(1, len(values) + 1)
    plt.plot(x, values, marker="o", markersize=3, linewidth=1, color="steelblue")

    colors = plt.cm.tab10.colors
    for i, e in enumerate(elbows):
        plt.axvline(e, color=colors[i % len(colors)], linestyle="--", alpha=0.7,
                    label=f"elbow {i+1}: {e}")
        if e - 1 < len(values):
            plt.scatter([e], [values[e - 1]], color=colors[i % len(colors)], zorder=5)

    if xlim:
        plt.xlim(0, xlim)

    plt.xlabel("Component index")
    plt.ylabel("Singular value")
    plt.title(f"M_joint singular values (full_rank={full_rank}, split used: r={r}, rfk={rfk}, rk={rk})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    suffix = "" if xlim is None else "_zoom"
    fname = f"m_joint_fullrank{full_rank}{suffix}.png"
    out_path = FIG_ROOT / fname
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"  elbow candidates: {elbows}")
    print(f"  saved: {out_path}")

    return elbows


# ----------------------------------------------------------------------
# Core computation
# ----------------------------------------------------------------------

def run_full_rank_scan(
    data: GjiveData,
    full_ranks: Sequence[int],
    group_assignments: Sequence[int],
    n_groups: int,
):
    K = data.A.shape[0]

    results = {}

    for full_rank in full_ranks:
        print(f"\n===== full_rank={full_rank} =====")

        r, rfk, rk = make_valid_split(full_rank)
        print(f"  using split: r={r}, rfk={rfk}, rk={rk}")

        rfk_list = [rfk] * n_groups
        rk_list = [rk] * K

        M = M_joint(
            data.A,
            r,
            rfk_list,
            rk_list,
            group_assignments,
        )

        # M_joint is a symmetric PSD averaged projection matrix, so its
        # eigenvalues (via eigvalsh) are the same as its singular values.
        singular_values = np.linalg.eigvalsh(M)[::-1]

        elbows = plot_scree(singular_values, full_rank, r, rfk, rk)
        plot_scree(singular_values, full_rank, r, rfk, rk, xlim=min(50, full_rank))

        results[full_rank] = {
            "singular_values": singular_values,
            "elbows": elbows,
            "r_estimate": elbows[0] if elbows else None,
        }

        print(f"  -> r estimate (first elbow): {results[full_rank]['r_estimate']}")

    return results


# ----------------------------------------------------------------------
# Main / CLI
# ----------------------------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)
    X, group_assignments = transform(df)
    data = GjiveData.from_real_data(X, group_assignments - 1)
    return data, group_assignments - 1


def main():
    parser = argparse.ArgumentParser(
        description="Estimate r from the M_joint singular value scree plot."
    )
    parser.add_argument(
        "--full-rank", type=int, nargs="+", required=True,
        help="One or more full_rank totals to test (r + rfk + rk == full_rank)."
    )
    parser.add_argument("--n-groups", type=int, default=2)
    parser.add_argument(
    "--stage", choices=["joint", "group"], default="joint",
    help="'joint' runs the r-estimation scan (stage 1). "
         "'group' runs the rfk-estimation scan (stage 2), requires --r and --full-rank."
)
    parser.add_argument("--r", type=int, default=None,
                        help="Fixed joint rank r, required for --stage group")
    args = parser.parse_args()

    data, group_assignments = load_data()

    if args.stage == "joint":
        run_full_rank_scan(data, args.full_rank, group_assignments, args.n_groups)

    elif args.stage == "group":
        if args.r is None:
            raise ValueError("--r is required for --stage group")
        for full_rank in args.full_rank:
            remaining = full_rank - args.r
            run_group_scan(data, args.r, remaining, group_assignments, args.n_groups)


if __name__ == "__main__":
    main()