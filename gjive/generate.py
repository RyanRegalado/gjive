from typing import Optional
import numpy as np

# Calculations
from .utils import (
    generate_random_matrix,
    orthonormalize,
    _validate_matrix_dims,
    check_orthogonal,
    to_object_array
)
from scipy.stats import ortho_group

# Data Handling
import json
from pathlib import Path
from .specifications import SimulationSpec
from .dataset import GjiveData
from dataclasses import asdict


def generate_joint(
    n: int,
    r: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a random n x r matrix with orthonormal columns.
    """

    if r > n:
        raise ValueError(f"r ({r}) cannot exceed n ({n})")

    Q_full = ortho_group.rvs(
        dim=n,
        random_state=seed,
    )

    return Q_full[:, :r]


def generate_group(
    U: np.ndarray,
    rfk: int,
    X: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate group-wise orthonormal basis Uf(k).
    """

    n, r = U.shape

    if rfk < 0:
        raise ValueError("rfk must be non-negative.")

    if rfk > n - r:
        raise ValueError(
            f"Requested group rank {rfk} exceeds dimension {n-r}."
        )

    P = np.eye(n) - U @ U.T

    return orthonormalize(
        P,
        X,
        n,
        rfk,
        seed=seed,
    )


def generate_ind(
    U: np.ndarray,
    Ufk: np.ndarray,
    rk: int,
    X: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate individual orthonormal basis Uk.
    """

    n, r = U.shape
    _, rfk = Ufk.shape

    check_orthogonal(
        U,
        Ufk,
        "U",
        "Ufk",
    )

    if rk < 0:
        raise ValueError("rk must be non-negative.")

    if rk > n - r - rfk:
        raise ValueError(
            f"Requested individual rank {rk} exceeds available dimension."
        )

    P = np.eye(n) - U @ U.T - Ufk @ Ufk.T

    return orthonormalize(
        P,
        X,
        n,
        rk,
        seed=seed,
    )


def generate_matrix(
    U: np.ndarray,
    Ufk: np.ndarray,
    Uk: np.ndarray,
    Vk: Optional[np.ndarray] = None,
    Wk: Optional[np.ndarray] = None,
    Xk: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    noise: Optional[int] = 0,
):
    """
    Construct Ak and return its loading matrices.
    """

    n = U.shape[0]

    rng = np.random.default_rng(seed)

    if n != Ufk.shape[0] or n != Uk.shape[0]:
        raise ValueError(
            "All matrices must have the same number of rows."
        )

    if Vk is None:
        Vk = generate_random_matrix(
            U.shape[1],
            symmetric=True,
            seed=seed,
        )

    if Wk is None:
        Wk = generate_random_matrix(
            Ufk.shape[1],
            symmetric=True,
            seed=seed,
        )

    if Xk is None:
        Xk = generate_random_matrix(
            Uk.shape[1],
            symmetric=True,
            seed=seed,
        )

    Ak = (
        U @ Vk @ U.T
        + Ufk @ Wk @ Ufk.T
        + Uk @ Xk @ Uk.T
    )

    _validate_matrix_dims(
        Ak,
        n,
        n,
    )

    if noise != 0:
        Ak += rng.standard_normal((n, n))

    return Ak, Vk, Wk, Xk


def generate_simulation_data(
    specs: SimulationSpec,
    simulation_name: str,
) -> GjiveData:
    """
    Generate a complete GJIVE simulation and save it.

    Saves:
        data/{simulation_name}/simulation.npz
        data/{simulation_name}/metadata.json
    """

    # -----------------------------
    # Joint subspace
    # -----------------------------

    U = generate_joint(
        n=specs.n,
        r=specs.r,
        seed=specs.seed,
    )


    # -----------------------------
    # Group subspaces
    # -----------------------------

    groups = np.unique(
        specs.group_assignment
    )

    Ufk = {}

    for i, group in enumerate(groups):

        Ufk[group] = generate_group(
            U=U,
            rfk=specs.rfk[group],
            seed=specs.seed + 100 + i,
        )


    # -----------------------------
    # Individual subspaces
    # -----------------------------

    Uk = []

    for k in range(specs.K):

        group = specs.group_assignment[k]

        Uk.append(
            generate_ind(
                U=U,
                Ufk=Ufk[group],
                rk=specs.rk[k],
                seed=specs.seed + 1000 + k,
            )
        )


    # -----------------------------
    # Generate A_k and loadings
    # -----------------------------

    A = []
    V = []
    W = []
    X = []

    for k in range(specs.K):

        group = specs.group_assignment[k]

        Ak, Vk, Wk, Xk = generate_matrix(
            U=U,
            Ufk=Ufk[group],
            Uk=Uk[k],
            seed=specs.seed + 10000 + k,
        )

        A.append(Ak)
        V.append(Vk)
        W.append(Wk)
        X.append(Xk)


    A = np.asarray(A)


    # -----------------------------
    # Storage
    # -----------------------------

    simulation_dir = (
        Path("data")
        / simulation_name
    )

    simulation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    Uf_array = to_object_array(
        [
            Ufk[g]
            for g in sorted(Ufk)
        ]
    )

    Uk_array = to_object_array(Uk)

    V_array = to_object_array(V)
    W_array = to_object_array(W)
    X_array = to_object_array(X)


    np.savez(
        simulation_dir / "simulation.npz",
        A=A,
        U=U,
        Uf=Uf_array,
        Uk=Uk_array,
        V=V_array,
        W=W_array,
        X=X_array,
    )

    metadata = asdict(specs)
    metadata["dataset_name"] = simulation_name

    with (
        simulation_dir / "metadata.json"
    ).open("w") as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )


    return GjiveData(simulation_dir)