from pathlib import Path

from gjive.dataset import GjiveData
from gjive.estimate import estimate_data
from gjive.generate import generate_simulation_data
from gjive.simulation_spec import SimulationSpec
import numpy as np


def main():
    # Load a generated dataset

    specs = SimulationSpec(
        n = 200,
        K = 1000,
        r = 2,
        rfk = [2,2],
        rk = [2] * 1000,
        p = 0.5,
        seed = 1,
    )

    

    data = generate_simulation_data(specs, "basic")

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

def basic_tests():
        # Load a generated dataset
    data = GjiveData(Path("data/simulation_test"))

    # Use the true ranks from the metadata
    estimate = estimate_data(
        data,
        r=data.metadata["r"],
        rfk=data.metadata["rfk"],
        rk=data.metadata["rk"],
    )
    



    print(estimate.metadata)
    assert estimate.metadata["dataset_name"] == "simulation_test"
    
    print(estimate.U.shape)

    print(len(estimate.Uf))
    for i, u in enumerate(estimate.Uf):
        print(f"Uf[{i}] =", u.shape)

    print(len(estimate.Uk))
    for i, u in enumerate(estimate.Uk):
        print(f"Uk[{i}] =", u.shape)

    for i in range(len(estimate.V)):
        print(
            estimate.V[i].shape,
            estimate.W[i].shape,
            estimate.X[i].shape,
        )


    assert estimate.U.ndim == 2
    assert len(estimate.Uk) == len(estimate.V)
    assert len(estimate.Uk) == len(estimate.W)
    assert len(estimate.Uk) == len(estimate.X)
    assert len(estimate.Uk) == len(estimate.metadata["group_assignment"])

    print("\n\nComparing Estimate and Generation Dimensions")

    
    ## Matching Dimensions of Generation and Estimate

    assert data.U.shape == estimate.U.shape
    print("PASSED: Generated U and Estimated U have the same shape")

    assert len(data.Uf) == len(estimate.Uf)
    print("PASSED: Same number of group subspaces")

    for g in range(len(data.Uf)):
        assert data.Uf[g].shape == estimate.Uf[g].shape
    print("PASSED: All generated and estimated Uf matrices have the same shape")

    assert len(data.Uk) == len(estimate.Uk)
    print("PASSED: Same number of individual subspaces")

    for k in range(len(data.Uk)):
        assert data.Uk[k].shape == estimate.Uk[k].shape
    print("PASSED: All generated and estimated Uk matrices have the same shape")

    assert len(data.V) == len(estimate.V)
    print("PASSED: Same number of joint loading matrices")


    assert len(data.W) == len(estimate.W)
    print("PASSED: Same number of group loading matrices")


    assert len(data.X) == len(estimate.X)
    print("PASSED: Same number of individual loading matrices")

    assert data.group_assignment.tolist() == estimate.metadata["group_assignment"]
    print("PASSED: Group assignments match")


    return None

if __name__ == "__main__":
    main()
    #basic_tests()