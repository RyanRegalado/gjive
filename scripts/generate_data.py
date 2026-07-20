import numpy as np

from gjive.generate import generate_group, generate_joint, generate_ind, generate_matrix
from gjive.utils import check_orthogonal, check_orthonormal

from pathlib import Path

from gjive.specifications import SimulationSpec
from gjive.generate import generate_simulation_data

def og_test() -> None:

    U = generate_joint(200, 3)
    Ufk = generate_group(U, 3)
    Uk = generate_ind(U, Ufk, 3)

    result = generate_matrix(U=U, Ufk=Ufk, Uk=Uk)
    
    check_orthogonal(U, Ufk, "U", "Ufk")
    print("Orthogonality check passed: U and Ufk are orthogonal.")

    check_orthogonal(U, Uk, "U", "Uk")
    print("Orthogonality check passed: U and Uk are orthogonal.")

    check_orthogonal(Uk, Ufk, "Uk", "Ufk")
    print("Orthogonality check passed: Uk and Ufk are orthogonal.")

    check_orthonormal(U, "U")
    print("Orthonormality check passed: U is orthonormal.")

    check_orthonormal(Ufk, "Ufk")
    print("Orthonormality check passed: Ufk is orthonormal.")

    check_orthonormal(Uk, "Uk")
    print("Orthonormality check passed: Uk is orthonormal.")

    return result

def main() -> None:
    specs = SimulationSpec(
    n=100,
    K=6,
    r=3,
    rfk=[3,3],
    rk=[3,2,3,3,3,2],
    p=0.5,
    seed=42,
    )

    # Generate simulation
    data = generate_simulation_data(
        specs=specs,
        simulation_name="simulation_test",
    )

    # -----------------------------
    # Basic information
    # -----------------------------
    print("Simulation generated successfully!\n")

    print("Shapes")
    print("------")
    print("A :", data.A.shape)
    print("U :", data.U.shape)
    print("Number of Uf ranks given", data.Uf.shape)
    print("Number of Uk ranks given", data.Uk.shape)

    print("\nMetadata")
    print("--------")
    print(data.metadata)

    # -----------------------------
    # Verify saved files exist
    # -----------------------------
    print("\nFiles")
    print("-----")
    print(data.path / "simulation.npz", (data.path / "simulation.npz").exists())
    print(data.path / "metadata.json", (data.path / "metadata.json").exists())

    # -----------------------------
    # Simple sanity checks
    # -----------------------------
    assert data.A.shape == (specs.K, specs.n, specs.n)
    print("PASSED: A has the correct shape")
    assert data.U.shape == (specs.n, specs.r)
    print("PASSED: U has the correct shape")

    for m, uf in enumerate(data.Uf):
        assert(uf.shape == (specs.n, specs.rfk[m]))
        print(f'PASSED: Uf[{m}] has the correct shape')


    for k, uk in enumerate(data.Uk):
        assert(uk.shape == (specs.n, specs.rk[k]))
        print(f'PASSED: Uk[{k}] has the correct shape')

    print("\nAll sanity checks passed.")

    return None

if __name__ == "__main__":
    main()
