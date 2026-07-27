import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gjive.estimate import estimate_data
from autism.utils import get_region
from gjive.estimate_class import GjiveEstimate
from gjive.estimate_spec import EstimateSpec
from gjive.dataset import GjiveData


DATA_PATH = Path.cwd() / "autism" / "csvs" / "processed_connectivity_dataset.csv"
ROI_PATH = Path().cwd() / "autism" / "csvs" / "CC200_ROI_labels.csv"
FIG_SAVE_PATH = Path().cwd() / "autism" / "figures"

# Shorthand code -> substring to match against region labels (case-insensitive)
REGION_SHORTHANDS = {
    "c": "Cerebellum",
    "t": "Temporal",
    "f": "Frontal",
    "p": "Parietal",
    "o": "Occipital",
    "l": "Limbic",
    "sc": "Subcortical",
    "in": "Insula",
    "cc": "Cingulate",
}


def resolve_region_filter(region_args: list[str] | None, all_labels: np.ndarray) -> list[str] | None:
    """
    Resolve --regions CLI input into a concrete list of region labels to display.

    Accepts shorthand codes (e.g. 'c' for Cerebellum, 't' for Temporal) via
    REGION_SHORTHANDS, or full region label strings matched case-insensitively
    as substrings against the actual labels present in the data.

    Returns None if region_args is None (meaning: show all regions).
    Raises ValueError if a requested code/name matches nothing.
    """
    if region_args is None:
        return None

    unique_labels = np.unique(all_labels)
    resolved = set()

    for arg in region_args:
        # strip a leading slash if the user typed it like "/c"
        key = arg.lstrip("/").strip()

        # try shorthand lookup first (case-insensitive)
        target = REGION_SHORTHANDS.get(key.lower(), key)

        matches = [lbl for lbl in unique_labels if target.lower() in lbl.lower()]

        if not matches:
            raise ValueError(
                f"No region labels matched '{arg}' (resolved to '{target}'). "
                f"Available labels: {sorted(unique_labels)}"
            )

        resolved.update(matches)

    return sorted(resolved)


def transform(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df_clean = df.dropna().copy()

    print("Original subjects:", len(df))
    print("Remaining subjects:", len(df_clean))

    dx_group = df_clean["DX_GROUP"].to_numpy()

    df_connectivity = df_clean.drop(columns=["SUB_ID", "DX_GROUP"])

    n_subjects = len(df_connectivity)
    n_regions = 200

    X = np.zeros((n_subjects, n_regions, n_regions), dtype=float)

    for col in df_connectivity.columns:
        region_a, region_b = map(int, col.split("_"))
        i = region_a - 1
        j = region_b - 1
        values = df_connectivity[col].to_numpy()
        X[:, i, j] = values
        X[:, j, i] = values

    for k in range(n_subjects):
        np.fill_diagonal(X[k], 1)

    print("Tensor shape:", X.shape)
    print(f'Group Assignments Length: {len(dx_group)}')
    return X, dx_group


def get_group_assignments(df: pd.DataFrame, group_col_name: str) -> np.ndarray:
    return df[group_col_name].to_numpy()


def get_regions_list():
    aals = pd.read_csv(ROI_PATH)['AAL'].tolist()
    regions = []
    for aal in aals:
        regions.append(get_region(aal))
    return regions


import plotly.graph_objects as go
import plotly.express as px

def visualize(
    estimate_matrix: np.ndarray,
    matrix_name: str,
    save_path: Path,
    cols: tuple[int, int, int] = (0, 1, 2),
    region_filter: list[str] | None = None,
    interactive: bool = False,
):
    n_cols = estimate_matrix.shape[1]

    for col in cols:
        if not 0 <= col < n_cols:
            raise ValueError(
                f"Column index {col} is out of bounds for an estimated matrix with {n_cols} columns."
            )

    coords = estimate_matrix[:, list(cols)]

    regions = get_regions_list()
    labels = np.asarray(regions)

    if region_filter is not None:
        keep_mask = np.isin(labels, region_filter)
        coords = coords[keep_mask]
        labels = labels[keep_mask]

        if len(labels) == 0:
            raise ValueError("region_filter excluded all points -- nothing to plot.")

    unique_labels = np.unique(labels)

    # ---------------------------------------------------------
    # Default: static matplotlib PNG (always produced)
    # ---------------------------------------------------------
    cmap = plt.colormaps["tab20"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, label in enumerate(unique_labels):
        color = cmap(i / max(1, len(unique_labels) - 1))
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            coords[mask, 2],
            color=color,
            label=label,
            s=50,
        )

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.set_title(f"Estimated Group Subspace: {matrix_name}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    save_path.mkdir(parents=True, exist_ok=True)
    png_path = save_path / f"{matrix_name}.png"
    plt.savefig(png_path)
    plt.close(fig)
    print(f"Visualization Success!\nPlot saved to {png_path}")

    # ---------------------------------------------------------
    # Optional: interactive Plotly HTML (only if requested)
    # ---------------------------------------------------------
    if interactive:
        color_map = px.colors.qualitative.Alphabet

        pfig = go.Figure()
        for i, label in enumerate(unique_labels):
            mask = labels == label
            pfig.add_trace(go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode="markers",
                marker=dict(size=5, color=color_map[i % len(color_map)]),
                name=label,
                text=[label] * mask.sum(),
                hoverinfo="text",
            ))

        pfig.update_layout(
            title=f"Estimated Group Subspace: {matrix_name}",
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3",
            ),
            legend=dict(itemsizing="constant"),
            width=1000,
            height=800,
        )

        html_path = save_path / f"{matrix_name}.html"
        pfig.write_html(html_path)
        print(f"Interactive plot saved to {html_path}")


def procrustes_transformation(
    to_align: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Align one estimated subspace to a reference subspace using
    an orthogonal Procrustes transformation.
    """
    to_align = np.asarray(to_align, dtype=float)
    reference = np.asarray(reference, dtype=float)

    if to_align.shape != reference.shape:
        raise ValueError(
            f"Matrices must have the same shape. "
            f"Received {to_align.shape} and {reference.shape}."
        )

    A = to_align.T @ reference
    V1, _, V2t = np.linalg.svd(A)
    O = V1 @ V2t

    return to_align @ O


def parse_args():
    parser = argparse.ArgumentParser(description="GJIVE subspace estimation + 3D visualization")

    parser.add_argument("--r", type=int, required=True,
                         help="Joint rank r")
    parser.add_argument("--rfk", type=int, nargs="+", required=True,
                         help="Group-specific ranks, one per group (e.g. --rfk 12 12)")
    parser.add_argument("--rk", type=int, required=True,
                         help="Individual rank rk, applied uniformly to all subjects")
    parser.add_argument("--cols", type=int, nargs=3, default=[0, 1, 2],
                         help="Zero-indexed columns of the estimated subspace to plot "
                              "as (x, y, z). Default: 0 1 2")
    parser.add_argument(
    "--regions", type=str, nargs="+", default=None,
    help="Subset of brain regions to display, by shorthand code (e.g. c=Cerebellum, "
         "t=Temporal, f=Frontal, p=Parietal, o=Occipital, l=Limbic, sc=Subcortical, "
         "in=Insula, cc=Cingulate) or full region name. Omit to show all regions."
    )

    parser.add_argument(
        "--interactive", action="store_true",
        help="Also generate an interactive 3D Plotly HTML plot alongside the default PNG."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(DATA_PATH)

    print(f"Transforming dataframe...")
    X, group_assignments = transform(df)
    K = X.shape[0]
    print(f"Dataframe Transformed!\nTransformed data shape: {X.shape}")

    data = GjiveData.from_real_data(X, group_assignments - 1)

    est_spec = EstimateSpec(
        r=args.r,
        rfk=args.rfk,
        rk=[args.rk] * K,
    )

    print(f'Estimating...')
    estimate = estimate_data(data, est_spec)
    print(f'Estimation success!')

    Uf0 = estimate.get_Uf(0)
    Uf1 = estimate.get_Uf(1)

    r = min(Uf0.shape[1], Uf1.shape[1])

    Uf0 = Uf0[:, :r]
    Uf1 = Uf1[:, :r]

    Uf1_tilde = procrustes_transformation(Uf1, Uf0)

    all_labels = np.asarray(get_regions_list())
    region_filter = resolve_region_filter(args.regions, all_labels)

    rfk_str = "_".join(str(v) for v in args.rfk)
    ranks_label = f"ranks_{args.r}_{rfk_str}_{args.rk}"
    cols_label = "_".join(str(c + 1) for c in args.cols)

    region_label = "all" if region_filter is None else "_".join(
        r.replace(" ", "") for r in region_filter
    )

    save_dir = FIG_SAVE_PATH / "3d" / ranks_label / region_label

    print(f'Visualizing...')
    visualize(
            Uf0,
            f"Uf_0_ASD_cols_{cols_label}",
            save_path=save_dir,
            cols=tuple(args.cols),
            region_filter=region_filter,
            interactive=args.interactive,
        )
    visualize(
        Uf1_tilde,
        f"Uf_1_control_transformed_cols_{cols_label}",
        save_path=save_dir,
        cols=tuple(args.cols),
        region_filter=region_filter,
        interactive=args.interactive,
    )
if __name__ == "__main__":
    main()