import numpy as np

from .utils import _as_2d_array, _validate_matrix_product, _validate_square_matrix, matrix_neg_half, orthonormalize


def U_joint(A, r, r_k):
    """Compute the joint signal subspace from a collection of matrices.
    
        Parameters:
            A (np.array) with shape (k, n, n). Data matrices.

            r (int). Estimated rank of U joint.

            r_k (list). Element r_k[i] is the estimated rank for data matrix i ( A[i] ).

        Returns:

            Joint subspace.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 3:
        raise ValueError(f"A must be a 3D array of shape (k, n, n). Received shape {A.shape}.")

    k, n, n2 = A.shape
    if n != n2:
        raise ValueError(f"Each matrix in A must be square. Received shape {A.shape}.")
    if len(r_k) != k:
        raise ValueError(f"Expected {k} ranks, received {len(r_k)}.")
    if r < 0:
        raise ValueError("r must be non-negative.")

    M = np.zeros((n, n), dtype=float)
    for i, matrix in enumerate(A):
        matrix = _validate_square_matrix(matrix, f"A[{i}]")
        eigvals, eigvecs = np.linalg.eigh(matrix)
        idx = np.argsort(eigvals)[::-1]
        rank = r + r_k[i]
        if rank > n:
            raise ValueError(f"Requested rank {rank} exceeds matrix dimension {n}.")
        Q_r = eigvecs[:, idx[:rank]]
        M += Q_r @ Q_r.T

    M /= k
    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx[:r]]


def U_group(A, U, r_fk, rk):
    """Compute the group-wise shared basis orthogonal to the joint basis.
    
        Parameters:
            U  is the joint subspace matrix.

            r_fk (int) is the estimated rank of the group wise structure.

            r_k (list of int). Where r_k[i] is the estimated rank for data matrix i. 

        Returns:
            
            U_group (np.array). 
    """

    

    return U_group


def U_ind(U, Ufk, rk, X=None):
    """Compute the individual basis orthogonal to both the joint and group subspaces."""
    U = _as_2d_array(U, "U")
    Ufk = _as_2d_array(Ufk, "Ufk")

    if U.shape[0] != Ufk.shape[0]:
        raise ValueError("U and Ufk must have the same number of rows.")
    \
    n = U.shape[0]
    
    if r < 0:
        raise ValueError("r must be non-negative.")
    if r > n:
        raise ValueError(f"Requested individual rank {rk} exceeds dimension {n}.")

    P = np.eye(n) - U @ U.T - Ufk @ Ufk.T
    
    U_ind = orthonormalize(P, X, n, rk)

    return  U_ind


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
