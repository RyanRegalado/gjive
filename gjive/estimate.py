from .utils import _as_2d_array, _validate_matrix_product, _validate_square_matrix, matrix_neg_half, orthonormalize

import numpy as np
from numpy.typing import NDArray
from typing import Sequence


def U_joint(
    A: NDArray[np.float64],
    r: int,
    rfk: Sequence[int],
    rk: Sequence[int],
    group_assignments: Sequence[int],
) -> NDArray[np.float64]:
    """
    Estimate the joint subspace U using the GJIVE joint projection step.

    For each matrix A_k, the top (r + r_f(k) + r_k) eigenvectors define the
    estimated signal space. The average projection matrix onto these spaces is
    then decomposed, and the top r eigenvectors give the estimated joint space.

    Parameters
    ----------
    A : np.ndarray, shape (K, n, n)
        Symmetric input matrices.
    r : int
        Estimated rank of the joint subspace.
    rfk : sequence of int
        Estimated group ranks, indexed by group assignment.
    rk : sequence of int
        Estimated individual ranks for each matrix.
    group_assignments : sequence of int
        Group label for each matrix.

    Returns
    -------
    np.ndarray, shape (n, r)
        Estimated joint subspace basis U.
    """

    A = np.asarray(A, dtype=float)

    if A.ndim != 3:
        raise ValueError(
            f"A must be a 3D array with shape (K, n, n). Received shape {A.shape}."
        )

    K, n, n2 = A.shape

    if n != n2:
        raise ValueError(
            f"Each matrix in A must be square. Received shape {A.shape}."
        )

    if not isinstance(r, int) or r < 0:
        raise ValueError("r must be a non-negative integer.")

    if len(rk) != K:
        raise ValueError(
            f"Expected {K} individual ranks, received {len(rk)}."
        )

    if len(group_assignments) != K:
        raise ValueError(
            f"Expected {K} group assignments, received {len(group_assignments)}."
        )

    if any(g < 0 or g >= len(rfk) for g in group_assignments):
        raise ValueError(
            "All group assignments must correspond to valid indices in rfk."
        )

    M = np.zeros((n, n), dtype=float)

    for i, ak in enumerate(A):
        group = group_assignments[i]

        eigvals, eigvecs = np.linalg.eigh(ak)
        idx = np.argsort(eigvals)[::-1]

        signal_rank = r + rfk[group] + rk[i]

        if signal_rank > n:
            raise ValueError(
                f"Requested signal rank {signal_rank} exceeds matrix dimension {n}."
            )

        Q = eigvecs[:, idx[:signal_rank]]

        M += Q @ Q.T

    M /= K

    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(eigvals)[::-1]

    return eigvecs[:, idx[:r]]


def U_group(
    A: NDArray[np.float64],
    U: NDArray[np.float64],
    rfk: Sequence[int],
    rk: Sequence[int],
    group_assignments: Sequence[int],
    group_id: int,
) -> NDArray[np.float64]:

    """
    Estimate the group-specific subspace U_f(g) in the GJIVE model.

    Removes the estimated joint space:
        B_k = (I - UU^T)A_k

    and applies AJIVE within the selected group using ranks
    r_f(g) and r_k.

    Returns the estimated group subspace basis.
    """

    A = np.asarray(A, dtype=float)
    U = np.asarray(U, dtype=float)

    if A.ndim != 3:
        raise ValueError("A must have shape (K,n,n).")

    K, n, n2 = A.shape

    if n != n2:
        raise ValueError("Matrices in A must be square.")

    if U.ndim != 2 or U.shape[0] != n:
        raise ValueError("U must have shape (n,r).")

    if len(rk) != K:
        raise ValueError("rk must contain one rank per matrix.")

    if len(group_assignments) != K:
        raise ValueError("group_assignments must match number of matrices.")

    if group_id >= len(rfk):
        raise ValueError("Invalid group_id.")

    group_idx = {
        i for i, g in enumerate(group_assignments)
        if g == group_id
    }

    if len(group_idx) == 0:
        raise ValueError(f"No matrices found in group {group_id}.")

    M = np.zeros((n, n), dtype=float)

    P = np.eye(n) - U @ U.T

    for i in group_idx:
        bk = P @ A[i]

        eigvals, eigvecs = np.linalg.eigh(bk)
        idx = np.argsort(eigvals)[::-1]

        signal_rank = rfk[group_id] + rk[i]

        if signal_rank > n:
            raise ValueError("Signal rank exceeds matrix dimension.")

        Q = eigvecs[:, idx[:signal_rank]]

        M += Q @ Q.T

    M /= len(group_idx)

    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(eigvals)[::-1]

    return eigvecs[:, idx[:rfk[group_id]]]



def U_ind():

    return None


def gjive(A, r, r_k, r_fk, r_ind, seed=None):
    """Run the full GJIVE factorization workflow."""
    rng = np.random.default_rng(seed)
    A = np.asarray(A, dtype=float)
    if A.ndim != 3:
        raise ValueError(f"A must be a 3D array of shape (k, n, n). Received shape {A.shape}.")

    k, n, n2 = A.shape
    if n != n2:
        raise ValueError(f"Each matrix in A must be square. Received shape {A.shape}.")
    if len(r_k) != k:
        raise ValueError(f"Expected {k} ranks, received {len(r_k)}.")
    if r > n:
        raise ValueError(f"Requested joint rank {r} exceeds dimension {n}.")
    if r_fk > n:
        raise ValueError(f"Requested group rank {r_fk} exceeds dimension {n}.")
    if r_ind > n:
        raise ValueError(f"Requested individual rank {r_ind} exceeds dimension {n}.")

    Uj = U_joint(A, r, r_k)
    Xg = rng.standard_normal((n, r_fk))
    Ug = U_group(Uj, r_fk, X=Xg)
    Xi = rng.standard_normal((n, r_ind))
    Ui = U_ind(Uj, Ug, r_ind, X=Xi)
    return Uj, Ug, Ui
