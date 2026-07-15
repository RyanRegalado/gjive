import numpy as np
from numpy.typing import NDArray
from typing import Optional


def scale_loadings(
    Vk: NDArray[np.float64],
    Wk: list[NDArray[np.float64]],
    Xk: list[NDArray[np.float64]],
    signal_scale: Optional[float] = 1.0,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """
    Scale loading matrices so the smallest singular value across all
    loading matrices is 1.

    Optional signal scaling parameter to increase the minimum singular value across all matrices.
    """

    singular_values = []

    singular_values.extend(np.linalg.svd(Vk, compute_uv=False))

    singular_values.extend(np.linalg.svd(Wk, compute_uv=False))

    singular_values.extend(np.linalg.svd(Xk, compute_uv=False))

    min_singular_value = np.min(singular_values)

    if min_singular_value <= 0:
        raise ValueError(
            f"Cannot scale: smallest singular value is {min_singular_value}"
        )

    scale = signal_scale / min_singular_value

    Vk = Vk * scale
    Wk = Wk * scale
    Xk = Xk * scale

    return Vk, Wk, Xk

def generate_noise(
    n: int,
    noise_scale: float,
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Generate symmetric Gaussian noise.

    Starts with iid N(0,1) entries:
        E = noise_scale * (Z + Z.T) / 2

    Result:
        diagonal ~ N(0, noise_scale^2)
        off-diagonal ~ N(0, noise_scale^2 / 2)
    """

    rng = np.random.default_rng(seed)

    Z = rng.normal(
        loc=0,
        scale=1,
        size=(n, n),
    )

    return noise_scale * (Z + Z.T) / 2
