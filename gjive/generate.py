from typing import Optional
import numpy as np

# Calculations

from .utils import generate_random_matrix, orthonormalize, _validate_matrix_dims, check_orthogonal, check_orthonormal
from scipy.stats import ortho_group

# Data Handling

import json
from pathlib import Path
from .specifications import SimulationSpec
from .dataset import GjiveData
from dataclasses import asdict # For JSON writing


def generate_joint(n: int, r: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random n x r matrix with orthonormal columns.

    Parameters
    ----------
    n : int
        Ambient dimension (number of rows).
    r : int
        Number of orthonormal columns (r <= n).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Q : np.ndarray, shape (n, r)
        Matrix with orthonormal columns (Q.T @ Q = I_r).
    """
    
    if r > n:
        raise ValueError(f"r ({r}) cannot exceed n ({n})")
    
    Q_full = ortho_group.rvs(dim=n, random_state=seed)

    return Q_full[:, :r]


def generate_group(
    U: np.ndarray,
    rfk: int,
    X: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a group-wise orthonormal basis Uf(k), orthogonal to U.

    The group-wise basis is constructed by projecting a (possibly random)
    matrix onto the orthogonal complement of U's column space and then
    orthonormalizing the result, ensuring Uf(k) is mutually orthogonal to U.

    Parameters
    ----------
    U : np.ndarray of shape (n, r)
        Joint/global orthonormal basis.
    rfk : int
        Rank of the group-wise structure for group f(k), i.e. number of
        columns of Uf(k).
    X : np.ndarray, optional
        Raw matrix to project and orthonormalize. If None, a random matrix
        is generated internally. Default is None.
    seed : int, optional
        Seed for reproducible random generation. Default is None.

    Returns
    -------
    np.ndarray of shape (n, rfk)
        Orthonormal matrix Uf(k) spanning the group-wise subspace, orthogonal
        to U.

    Raises
    ------
    ValueError
        If rfk is negative, or if rfk exceeds the dimension of the
        orthogonal complement of U (n - r).
    """
    n, r = U.shape

    if rfk < 0:
        raise ValueError("rfk must be non-negative.")
    if rfk > n - r:
        raise ValueError(f"Requested group rank {rfk} exceeds dimension {n}.")

    P = np.eye(n) - U @ U.T

    U_group = orthonormalize(P, X, n, rfk, seed=seed)

    return U_group


def generate_ind(
    U: np.ndarray,
    Ufk: np.ndarray,
    rk: int,
    X: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate an individual orthonormal basis Uk, orthogonal to both U and Uf(k).

    The individual basis is constructed by projecting a (possibly random)
    matrix onto the orthogonal complement of the joint span of U and Uf(k),
    and then orthonormalizing the result.

    Parameters
    ----------
    U : np.ndarray of shape (n, r)
        Joint/global orthonormal basis.
    Ufk : np.ndarray of shape (n, rfk)
        Group-wise orthonormal basis for observation k's group, must be
        mutually orthogonal to U.
    rk : int
        Rank of the individual structure for observation k, i.e. number of
        columns of Uk.
    X : np.ndarray, optional
        Raw matrix to project and orthonormalize. If None, a random matrix
        is generated internally. Default is None.
    seed : int, optional
        Seed for reproducible random generation. Default is None.

    Returns
    -------
    np.ndarray of shape (n, rk)
        Orthonormal matrix Uk spanning the individual subspace, orthogonal
        to both U and Ufk.

    Raises
    ------
    ValueError
        If U and Ufk do not have the same number of rows, if rk is negative,
        or if rk exceeds the dimension of the remaining orthogonal
        complement (n - r - rfk).
    AssertionError
        If U and Ufk are not mutually orthogonal.
    """
    n, r = U.shape
    _, rfk = Ufk.shape

    check_orthogonal(U, Ufk, "U", "Ufk")

    if U.shape[0] != Ufk.shape[0]:
        raise ValueError('Joint and Group-Wise matrices must have the same number of rows')

    if rk < 0:
        raise ValueError("rk must be non-negative.")
    if rk > n - r - rfk:
        raise ValueError(f"Requested group rank {rk} exceeds dimension {n}.")

    P = np.eye(n) - U @ U.T - Ufk @ Ufk.T

    Uk = orthonormalize(P, X, n, rk, seed=seed)

    return Uk


def generate_matrix(
    U: np.ndarray,
    Ufk: np.ndarray,
    Uk: np.ndarray,
    Vk: Optional[np.ndarray] = None,
    Wk: Optional[np.ndarray] = None,
    Xk: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    noise: Optional[int] = 0
) -> np.ndarray:
    """Construct a single observation Ak from joint, group, and individual structure.

    Builds Ak according to the three-term decomposition::

        Ak = U @ Vk @ U.T + Ufk @ Wk @ Ufk.T + Uk @ Xk @ Uk.T

    where U, Ufk, and Uk are mutually orthogonal orthonormal bases
    representing the joint, group-wise, and individual subspaces
    respectively, and Vk, Wk, Xk are symmetric coefficient matrices.

    Parameters
    ----------
    U : np.ndarray of shape (n, r)
        Joint/global orthonormal basis.
    Ufk : np.ndarray of shape (n, rfk)
        Group-wise orthonormal basis for observation k's group.
    Uk : np.ndarray of shape (n, rk)
        Individual orthonormal basis for observation k.
    Vk : np.ndarray of shape (r, r), optional
        Symmetric coefficient matrix for the joint component. If None, a
        random symmetric matrix is generated. Default is None.
    Wk : np.ndarray of shape (rfk, rfk), optional
        Symmetric coefficient matrix for the group-wise component. If None,
        a random symmetric matrix is generated. Default is None.
    Xk : np.ndarray of shape (rk, rk), optional
        Symmetric coefficient matrix for the individual component. If None,
        a random symmetric matrix is generated. Default is None.
    seed : int, optional
        Seed for reproducible random generation of Vk, Wk, and/or Xk when
        they are not provided. Default is None.

    Returns
    -------
    np.ndarray of shape (n, n)
        The constructed observation matrix Ak.

    Raises
    ------
    ValueError
        If U, Ufk, and Uk do not all have the same number of rows.
    """
    n = U.shape[0]

    rng = np.random.default_rng(seed)

    if n != Ufk.shape[0] or n != Uk.shape[0] or Ufk.shape[0] != Uk.shape[0]:
        raise ValueError("All matrices must have the same number of rows")

    if Vk is None:
        joint_rank_n = U.shape[1]
        Vk = generate_random_matrix(joint_rank_n, symmetric=True, seed=seed)

    if Wk is None:
        group_rank_n = Ufk.shape[1]
        Wk = generate_random_matrix(group_rank_n, symmetric=True, seed=seed)

    if Xk is None:
        ind_rank_n = Uk.shape[1]
        Xk = generate_random_matrix(ind_rank_n, symmetric=True, seed=seed)

    Ak = (U @ Vk @ U.T) + (Ufk @ Wk @ Ufk.T) + (Uk @ Xk @ Uk.T)

    _validate_matrix_dims(Ak, n, n)

    if noise != 0:
        Ak += rng.standard_normal((n, n))

    return Ak


def generate_simulation_data(
    specs: SimulationSpec,
    simulation_name: str,
) -> GjiveData:
    """
    Generate a complete GJIVE simulation and save it.

    Parameters
    ----------
    specs : SimulationSpec
        Simulation configuration.
    simulation_name : str
        Name of simulation file (without .npz extension).

    Returns
    -------
    GjiveData
        GjiveData object containing relevant matrix information and metadata.

    Saves
    -----
    data/{simulation_name}.npz
    """

    # -----------------------------
    # Generate joint subspace
    # -----------------------------
    U = generate_joint(
        n=specs.n,
        r=specs.r,
        seed=specs.seed,
    )

    # -----------------------------
    # Generate group-wise subspaces
    # -----------------------------
    groups = np.unique(specs.group_assignment)

    Ufk = {}

    for i, group in enumerate(groups):
        Ufk[group] = generate_group(
            U=U,
            rfk=specs.rfk[group],
            seed=specs.seed + 100 + i,
        )

    # -----------------------------
    # Generate individual subspaces
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
    # Generate matrices A_k
    # -----------------------------
    A = []

    for k in range(specs.K):
        group = specs.group_assignment[k]

        Ak = generate_matrix(
            U=U,
            Ufk=Ufk[group],
            Uk=Uk[k],
            seed=specs.seed + 10000 + k,
        )

        A.append(Ak)

    A = np.array(A)

    simulation_dir = Path("data") / simulation_name
    simulation_dir.mkdir(parents=True, exist_ok=True)

# ----- Changes made here to support variable-sized ranks for Uf and Uk.
# ----- A normal np.array() would attempt to create a rectangular 3D array
# ----- (K, n, r), requiring every rank r to be identical.
# ----- Object arrays allow each subspace matrix to have its own number of columns.

# Convert group dictionary to object array
# Ufk is stored as {group_id: group_subspace}, so we explicitly preserve
# the group ordering when converting to an array.
    Uf_array = np.empty(len(Ufk), dtype=object)

    for i, g in enumerate(sorted(Ufk)):
        Uf_array[i] = Ufk[g]

# Same fix for Uk

    Uk_array = np.empty(len(Uk), dtype=object)

    for i, uk in enumerate(Uk):
        Uk_array[i] = uk

    np.savez(
        simulation_dir / "simulation.npz",
        A=np.asarray(A),
        U=U,
        Uf=Uf_array,
        Uk=Uk_array,
    )

    with (simulation_dir / "metadata.json").open("w") as f:
        json.dump(asdict(specs), f, indent=4)

    return GjiveData(simulation_dir)