import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from gjive.estimate import estimate_data
from autism.utils import get_elbows, get_region
from gjive.estimate_class import GjiveEstimate
from gjive.estimate_spec import EstimateSpec
from gjive.dataset import GjiveData


DATA_PATH = Path.cwd() / "autism" / "csvs" / "processed_connectivity_dataset.csv"

ROI_PATH = Path().cwd() / "autism" / "csvs" / "CC200_ROI_labels.csv"

FIG_SAVE_PATH = Path().cwd() / "autism" / "figures"



def transform(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:

    # -------------------------------
    # 1. Remove subjects with NaNs
    # -------------------------------

    df_clean = df.dropna().copy()

    print("Original subjects:", len(df))
    print("Remaining subjects:", len(df_clean))


    # -------------------------------
    # 2. Extract metadata
    # -------------------------------

    dx_group = df_clean["DX_GROUP"].to_numpy()


    # Remove non-connectivity columns
    df_connectivity = df_clean.drop(
        columns=["SUB_ID", "DX_GROUP"]
    )

    # -------------------------------
    # 3. Create tensor
    # -------------------------------

    n_subjects = len(df_connectivity)
    n_regions = 200

    X = np.zeros(
        (n_subjects, n_regions, n_regions),
        dtype=float
    )


    # -------------------------------
    # 4. Fill upper triangle
    # -------------------------------

    for col in df_connectivity.columns:

        # Column name looks like "1_2"
        region_a, region_b = map(
            int,
            col.split("_")
        )

        # Convert 1-indexed ROI labels to numpy indices
        i = region_a - 1
        j = region_b - 1

        values = df_connectivity[col].to_numpy()

        # Fill both halves because correlation matrices are symmetric
        X[:, i, j] = values
        X[:, j, i] = values


    # -------------------------------
    # 5. Set diagonal
    # -------------------------------

    for k in range(n_subjects):
        np.fill_diagonal(X[k], 1)


    print("Tensor shape:", X.shape)
    print(f'Group Assignments Length: {len(dx_group)}')
    return X, dx_group

def get_group_assignments(df: pd.DataFrame, group_col_name: str) -> np.ndarray:

    return df[group_col_name].to_numpy() 

def get_regions_list():
    # Because matrix operations in the estimation algorithm DO NOT swap rows, we can keep this vector consistent.
    aals = pd.read_csv(ROI_PATH)['AAL'].tolist()

    regions = []

    for aal in aals:
        regions.append(get_region(aal))

    return regions

def find_ranks():

    return None

def visualize(estimate_matrix: np.ndarray,
              matrix_name: str,
              save_path: Path,
              cols: tuple[int, int, int] = (0,1,2)):

    n_cols = estimate_matrix.shape[1]

    for col in cols:
        if not 0 <= col < n_cols:
            raise ValueError(
                f"Column index {col} is out of bounds for an estimated matrix with {n_cols} columns."
            )

    coords = estimate_matrix[:,list(cols)]

    regions = get_regions_list()

    # Unique brain region/network labels
    labels = np.asarray(regions)
    unique_labels = np.unique(labels)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.colormaps["tab20"]

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
    fig_path = f"{matrix_name}_3d.png"
    plt.savefig(save_path / fig_path)
    print(f"Visualization Success!\nPlot saved to {save_path / fig_path}")

def main():

    df = pd.read_csv(DATA_PATH)

    print(f"Transforming dataframe...")
    X, group_assignments = transform(df)
    K = X.shape[0]
    print(f"Dataframe Transformed!\nTransformed data shape: {X.shape}")
    
    data = GjiveData.from_real_data(X, group_assignments - 1) # Transforming [2, 1, 1] -> [1, 0, 0] for consistency

    est_spec = EstimateSpec(
        r = 3,
        rfk = [3,3],
        rk = [3] * K
    )

    print(f'Estimating...')
    estimate = estimate_data(data, est_spec)
    print(f'Estimation success!')

    Uf0 = estimate.get_Uf(0)
    Uf1 = estimate.get_Uf(1)
    Uf1_tilde = procrustes_transformation(Uf1, Uf0)

    print(f'Visualizing...')
    visualize(Uf0, "Uf0_ASD", save_path=FIG_SAVE_PATH)
    visualize(Uf1_tilde, "Uf1_control_transformed", save_path=FIG_SAVE_PATH)

import numpy as np


def procrustes_transformation(
    to_align: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Align one estimated subspace to a reference subspace using
    an orthogonal Procrustes transformation.

    Computes:
        A = to_align.T @ reference

        A = V1 Lambda V2.T

        O = V1 @ V2.T

    and returns:
        aligned = to_align @ O

    Parameters
    ----------
    to_align : np.ndarray, shape (n, r)
        Estimated subspace to rotate.

    reference : np.ndarray, shape (n, r)
        Reference subspace.

    Returns
    -------
    np.ndarray, shape (n, r)
        Rotationally aligned version of `to_align`.
    """

    to_align = np.asarray(to_align, dtype=float)
    reference = np.asarray(reference, dtype=float)

    if to_align.shape != reference.shape:
        raise ValueError(
            f"Matrices must have the same shape. "
            f"Received {to_align.shape} and {reference.shape}."
        )

    # Compute A = Uf2.T @ Uf1
    A = to_align.T @ reference

    # SVD: A = V1 Lambda V2.T
    V1, _, V2t = np.linalg.svd(A)

    # O = V1 V2.T
    O = V1 @ V2t

    # Align Uf2
    return to_align @ O


if __name__ == "__main__":
    main()








