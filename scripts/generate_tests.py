import numpy as np
from pathlib import Path
from gjive.simulation_spec import SimulationSpec
from gjive.generate import generate_simulation_data
from gjive.dataset import GjiveData
from gjive.utils import check_orthogonal, check_orthonormal



def main() -> None:
    make_repros(1,3)
    
    print(f"\n\nSeed Uniqueness Test:")
    # Tests Reproducibility via seed
    A = GjiveData(Path("data/tests") / "repro_1")
    B = GjiveData(Path("data/tests") / "repro_2")
    C = GjiveData(Path("data/tests") / "repro_3")
    
    
    assert(A == B)
    print("PASSED: Datasets A and B are identical.")
    assert(A != C)
    assert(B != C)
    print("PASSED: Dataset C is not equal to A or B.")

    print("\n\nOrthogonal and Orthonormal Checks:")

    for k, _ in enumerate(A.A):
        group = A.get_group(k)
        print(f'Matrix A[{k}]')
        check_orthonormal(A.U)
        check_orthonormal(A.Uf[group])
        check_orthonormal(A.Uk[k])
        print(f'PASSED: All matrices from decomposition are orthonormal for A[{k}]')
        check_orthogonal(A.U, A.Uf[group], 'U', f'Uf[{group}]')
        check_orthogonal(A.U, A.Uk[k], 'U', f'Uk[{k}]')
        check_orthogonal(A.Uf[group], A.Uk[k], f'Uf[{group}]', f'Uk[{k}]')
        print(f'PASSED: All matrices from decomposition are orthogonal for A[{k}]')
        print("------")

    return None

def make_repros(seed1, seed2):
    """
    Creates two identical datasets and one different one.

    Returns (A, B, C) where A == B and (B != C & A != C)
    """
    specs1 = SimulationSpec(
    n=100,
    K=6,
    r=3,
    rfk=[3,3],
    rk=[3] * 6,
    p=0.5,
    seed=seed1,
    )
    specs2 = SimulationSpec(
    n=100,
    K=6,
    r=3,
    rfk=[3,3],
    rk=[3] * 6,
    p=0.5,
    seed=seed1,
    )
    
    specs3 = SimulationSpec(
    n=100,
    K=6,
    r=3,
    rfk=[3,3],
    rk=[3] * 6,
    p=0.5,
    seed=seed2,
    )

    data1 = generate_simulation_data(
        specs=specs1,
        simulation_name="tests/repro_1",
    )
    data2 = generate_simulation_data(
        specs = specs2,
        simulation_name="tests/repro_2"
    )

    data3 = generate_simulation_data(
        specs = specs3,
        simulation_name="tests/repro_3"
    )

    return data1, data2, data3


if __name__ == "__main__":
    main()