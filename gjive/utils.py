import numpy as np


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
