import numpy as np
from numpy.typing import NDArray
from typing import Sequence
from .dataset import GjiveData
from .estimate_class import GjiveEstimate
from .utils import to_object_array
#from irlb import irlb
from time import perf_counter


def U_joint(
    A: NDArray[np.float64],
    r: int,
    rfk: Sequence[int],
    rk: Sequence[int],
    group_assignments: Sequence[int],
) -> NDArray[np.float64]:
    """
    Estimate the joint subspace U using the GJIVE joint projection step.

    For each matrix A_k, the top (r + r_f(k) + r_k) leading left singular vectors define the
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

    #start = perf_counter()

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

        U, _ , _ = np.linalg.svd(ak, full_matrices=False)
        #Q, s, _, _, _= irlb(ak, signal_rank) Commented out, since its slower.
        Q = U[:, :signal_rank]


        M += Q @ Q.T

    M /= K

    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(eigvals)[::-1]

    #elapsed = perf_counter() - start
    #print(f'Elapsed time: {elapsed}')
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

        Uk, _, _= np.linalg.svd(bk, full_matrices=False)
        

        signal_rank = rfk[group_id] + rk[i]

        if signal_rank > n:
            raise ValueError("Signal rank exceeds matrix dimension.")

        Q = Uk[:, :signal_rank]

        M += Q @ Q.T

    M /= len(group_idx)

    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(eigvals)[::-1]

    return eigvecs[:, idx[:rfk[group_id]]]


def U_ind(
    Ak: NDArray[np.float64],
    U: NDArray[np.float64],
    Ufk: NDArray[np.float64],
    rk: int,
) -> NDArray[np.float64]:
    """
    Estimate the individual subspace U_k for one matrix in the GJIVE model.

    Removes the estimated joint and group contributions:

        R_k = I - U V_k^T - U_f W_k^T

    and extracts the top rk left singular vectors of the residual.

    Parameters
    ----------
    Ak : np.ndarray, shape (n,n)
        Data matrix.
    U : np.ndarray
        Estimated joint basis.
    Ufk : np.ndarray
        Estimated group basis.
    rk : int
        Individual rank.

    Returns
    -------
    np.ndarray
        Estimated individual basis U_k.
    """

    Ak = np.asarray(Ak, dtype=float)

    n = Ak.shape[0]

    if Ak.shape[0] != Ak.shape[1]:
        raise ValueError("Ak must be square.")

    if rk < 0 or rk > n:
        raise ValueError(
            f"Invalid individual rank {rk} for dimension {n}."
        )

    residual = (np.eye(n) - U @ U.T - Ufk @ Ufk.T) @ Ak

    Uk, _, _ = np.linalg.svd(residual, full_matrices=False)

    return Uk[:, :rk]

def estimate_loadings(Ak: NDArray[np.float64], 
                     U_hat: NDArray[np.float64],
                     Uf_hat: NDArray[np.float64],
                     Uk_hat: NDArray[np.float64]) -> NDArray[np.float64]:
    
    return ((Ak.T @ U_hat), (Ak.T @ Uf_hat), (Ak.T @ Uk_hat))


from pathlib import Path
import json
import numpy as np

...

def estimate_data(
    data: GjiveData,
    r: int,
    rfk: Sequence[int],
    rk: Sequence[int],
) -> GjiveEstimate:

    A = data.A
    group_assignments = data.group_assignment

    estimate_name = data.metadata["dataset_name"]

    U_hat = U_joint(A, r, rfk, rk, group_assignments)

    Uf_hat = []
    for group_id in range(len(set(group_assignments))):
        ufk_hat = U_group(A, U_hat, rfk, rk, group_assignments, group_id)
        Uf_hat.append(ufk_hat)

    Uk_hat = []
    for k, ak in enumerate(A):
        group = group_assignments[k]
        uk = U_ind(ak, U_hat, Uf_hat[group], rk[k])
        Uk_hat.append(uk)

    Vk_hat = []
    Wk_hat = []
    Xk_hat = []

    for k, ak in enumerate(A):
        group = group_assignments[k]
        vk, wk, xk = estimate_loadings(
            ak,
            U_hat,
            Uf_hat[group],
            Uk_hat[k],
        )
        Vk_hat.append(vk)
        Wk_hat.append(wk)
        Xk_hat.append(xk)

    Uf_array = to_object_array(Uf_hat)
    Uk_array = to_object_array(Uk_hat)

    V_array = to_object_array(Vk_hat)
    W_array = to_object_array(Wk_hat)
    X_array = to_object_array(Xk_hat)

    # ------------------------------------------------------------------
    # Write estimate to disk
    # ------------------------------------------------------------------

    cwd = Path().cwd()
    estimate_dir = cwd / "estimates" / estimate_name
    estimate_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        estimate_dir / "estimate.npz",
        U=U_hat,
        Uf=Uf_array,
        Uk=Uk_array,
        V=V_array,
        W=W_array,
        X=X_array,
    )

    metadata = {
        "dataset_name": estimate_name,
        "r": int(r),
        "rfk": list(rfk),
        "rk": list(rk),
        "group_assignment": data.group_assignment.tolist()
    }

    with (estimate_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=4)

    return GjiveEstimate(estimate_dir)