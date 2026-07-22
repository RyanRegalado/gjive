from pathlib import Path

from gjive.estimate import GjiveData
from gjive.estimate import estimate_data
from gjive.generate import generate_simulation_data
from gjive.simulation_spec import SimulationSpec
import numpy as np


def main():
    # Load a generated dataset

    specs = SimulationSpec(
        n = 15,
        K = 10,
        r = 2,
        rfk = [2,2],
        rk = [2] * 10,
        p = 0.5,
        seed = 1,
        noise = 10e-6,
        signal_scale= 10e6,
    )    

    data = generate_simulation_data(specs, "basic_noise")

    # Use the true ranks from the metadata
    estimate = estimate_data(
        data,
        r=data.metadata["r"],
        rfk=data.metadata["rfk"],
        rk=data.metadata["rk"],
    )

    print("Estimation vs. Generated Data")

    U = data.U
    U_hat = estimate.U
    Uf = data.Uf
    Uf_hat = estimate.Uf
    Uk = data.Uk
    Uk_hat = estimate.Uk

    tol = 1e-10

    print("U vs U_hat")
    U_error = np.linalg.norm(U @ U.T - U_hat @ U_hat.T)
    assert U_error < tol, f"FAILED: U subspace error too large ({U_error})"
    print("PASSED: U subspace recovered")


    print("\nUf vs Uf_hat")
    for g in range(len(Uf)):
        Uf_error = np.linalg.norm(
            Uf[g] @ Uf[g].T -
            Uf_hat[g] @ Uf_hat[g].T
        )
        assert Uf_error < tol, f"FAILED: Uf[{g}] subspace error too large ({Uf_error})"
        print(f"PASSED: Uf[{g}] subspace recovered")


    print("\nUk vs Uk_hat")
    for k in range(len(Uk)):
        Uk_error = np.linalg.norm(
            Uk[k] @ Uk[k].T -
            Uk_hat[k] @ Uk_hat[k].T
        )
        assert Uk_error < tol, f"FAILED: Uk[{k}] subspace error too large ({Uk_error})"
        print(f"PASSED: Uk[{k}] subspace recovered")
        
    return None

if __name__ == "__main__":
    main()