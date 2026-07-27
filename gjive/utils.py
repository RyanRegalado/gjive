import numpy as np
from typing import Sequence, Any
from irlb import irlb
from numpy.typing import NDArray


def _as_2d_array(array, name):
    arr = np.asarray(array, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array. Received shape {arr.shape}.")
    return arr


def _validate_square_matrix(matrix, name):
    arr = _as_2d_array(matrix, name)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square. Received shape {arr.shape}.")
    return arr


def _validate_matrix_product(left, right, left_name, right_name):
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            f"Cannot multiply {left_name} of shape {left.shape} with {right_name} of shape {right.shape}."
        )


def generate_random_matrix(n, m=None, symmetric=False, seed=None):

    rng = np.random.default_rng(seed)

    if symmetric:
        # Force square matrix
        M = rng.standard_normal((n, n))
        return (M + M.T) / 2

    # Non-symmetric case
    if m is None:
        raise ValueError("m must be provided when symmetric=False.")
    
    return rng.standard_normal((n, m))



def matrix_neg_half(matrix):
    """Compute the inverse square root of a symmetric positive semidefinite matrix."""

    matrix = _validate_square_matrix(matrix, "matrix")
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-12, None)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def _validate_matrix_dims(matrix, n, m, name="matrix"):

    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D. Got {matrix.ndim}D.")

    rows, cols = matrix.shape

    if n is not None and rows != n:
        raise ValueError(f"{name} must have {n} rows. Got {rows}.")

    if m is not None and cols != m:
        raise ValueError(f"{name} must have {m} columns. Got {cols}.")

    return matrix

def orthonormalize(P, X, rows, rank, seed=None):

    if X is None:
        X = generate_random_matrix(rows, rank, seed=seed)
    else:
        X = _as_2d_array(X, "X")

    _validate_matrix_dims(X, rows, rank)

    S, _, _ = np.linalg.svd(X, full_matrices=False)
    
    LHS = P @ S
    
    inner = S.T @ P @ S

    RHS = matrix_neg_half(inner)

    out = LHS @ RHS
    
    _validate_matrix_dims(out, rows, rank)

    return out



def check_orthogonal(A, B, nameA="__", nameB="__"):
    """
    Verify that two matrices have mutually orthogonal columns.

    Parameters
    ----------
    A : np.ndarray
    B : np.ndarray
    nameA : str, optional
        Label for A, used in the error message.
    nameB : str, optional
        Label for B, used in the error message.

    Raises
    ------
    ValueError
        If A @ B.T is not close to zero.
    """
    if not np.allclose(A.T @ B, 0, atol=1e-8):
        raise ValueError(f"{nameA} and {nameB} must be mutually orthogonal")


def check_orthonormal(A, name="__") -> bool:
    """
    Verify that A has orthonormal columns (A.T @ A == I).

    Parameters
    ----------
    A : np.ndarray
    name : str, optional
        Label for A, used in the error message.

    Raises
    ------
    ValueError
        If A.T @ A is not close to the identity matrix.
    """
    r = A.shape[1]
    if not np.allclose(A.T @ A, np.eye(r), atol=1e-8):
        raise ValueError(f"{name} must have orthonormal columns")
    

def to_object_array(items):
    """
    Convert variable-sized matrices into numpy object array.
    """

    arr = np.empty(
        len(items),
        dtype=object,
    )

    for i, item in enumerate(items):
        arr[i] = item

    return arr



def average_projection_matrix(
    matrices: NDArray[np.float64],
    r: int,
    rfk: int,
    rk: int,
    use_irlb: bool = True,
) -> NDArray[np.float64]:
    """
    Compute the average projection matrix

        M = (1/K) Σ Q_k Q_k^T,

    where Q_k contains the leading left singular vectors of the
    corresponding matrix.

    Signal ranks list is as follows: [r, rfk, r]
    """

    K, n, _ = matrices.shape

    M = np.zeros((n, n), dtype=float)

    signal_rank = r + rfk + rk

    for matrix in matrices:

        if signal_rank > n:
            raise ValueError(
                f"Requested signal rank {signal_rank} exceeds matrix dimension {n}."
            )

        if use_irlb:
            Q, _, _, _, _ = irlb(matrix, signal_rank)
        else:
            U, _, _ = np.linalg.svd(matrix, full_matrices=False)
            Q = U[:, :signal_rank]

        M += Q @ Q.T

    return M / K





def M_joint(
    A: NDArray[np.float64],
    r: int,
    rfk: Sequence[int],
    rk: Sequence[int],
    group_assignments: Sequence[int],
    use_irlb: bool = True,
) -> NDArray[np.float64]:


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

        signal_rank = r + rfk[group] + rk[i]

        if signal_rank > n:
            raise ValueError(
                f"Requested signal rank {signal_rank} exceeds matrix dimension {n}."
        )
        
        if use_irlb:
            Q, _, _, _, _= irlb(ak, signal_rank)
        else:
            U, _ , _ = np.linalg.svd(ak, full_matrices=False)
            Q = U[:, :signal_rank]

        M += Q @ Q.T

    M /= K

    return M




def M_group(
    A: NDArray[np.float64],
    U: NDArray[np.float64],
    rfk: Sequence[int],
    rk: Sequence[int],
    group_assignments: Sequence[int],
    group_id: int,
    use_irlb: bool = True,
) -> NDArray[np.float64]:

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

        signal_rank = rfk[group_id] + rk[i]

        if signal_rank > n:
            raise ValueError("Signal rank exceeds matrix dimension.")
        
        if use_irlb:
            Q, _, _, _, _= irlb(bk, signal_rank)
        else:
            Uk, _, _= np.linalg.svd(bk, full_matrices=False)
            Q = Uk[:, :signal_rank]
        
        M += Q @ Q.T

    M /= len(group_idx)

    return M
