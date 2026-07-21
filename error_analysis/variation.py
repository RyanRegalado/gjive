import numpy as np
import pandas as pd
from time import perf_counter
from pathlib import Path
# GJIVE Functions
from gjive.generate import generate_simulation_data
from gjive.estimate import estimate_data
# Classes
from gjive.dataset import GjiveData
from gjive.estimate_class import GjiveEstimate
from gjive.specifications import SimulationSpec

def generate_variation_K(base: SimulationSpec, k_values: list, name: str, track_time: bool = True):
    simulations = []
    times = {}

    output_path = Path().cwd() / "data" / name

    for i, k in enumerate(k_values):
        k = int(k)

        current_spec = SimulationSpec(
            n = base.n,
            K = k,
            r = base.r,
            rfk = [base.rfk[0]] * (len(base.rfk)),
            rk = [base.rk[0]] * k,
            p = base.p,
            seed = base.seed,
            signal_scale=base.signal_scale,
            noise = base.noise,
        )

        if track_time:
            t0 = perf_counter()
        
        sim_number = f'sim_{k}'
        sim = generate_simulation_data(current_spec, sim_number, output_path)
        simulations.append(sim)
        
        if track_time:
            elapsed = perf_counter() - t0
            times[int(k)] = elapsed
            print(f'Simulation {i} with K = {k} completed in: \n{round(elapsed, 10)} seconds\n')
    
    return simulations, times


def estimate_variation_K(datasets: list[GjiveData], name: str, track_time = True):
    estimates = []
    times = []
    length = len(datasets)

    output_path = Path().cwd() / "estimates" / name

    for i, data in enumerate(datasets):
        print(f"\nEstimating... ({i + 1} / {length}) \n")
        t0 = perf_counter()
        estimates.append(estimate_data(data, data.metadata["r"], 
                              data.metadata["rfk"],
                              data.metadata["rk"],
                              output_path / f'est_{data.metadata["K"]}'))
        if track_time:
            elapsed = perf_counter() - t0
            print(f'K = {data.metadata["K"]} completed in:\n{round(elapsed, 10)} seconds\n')
            times.append(elapsed) 
    return estimates, times

def K_vec(start, n, step):
    
    if start < 10:
        raise Warning(f'Start value of {start} is too small and may not form two distinct groups')
    
    k_values = np.arange(start, step * (n + 1), step)

    return k_values


def subspace_frob_norm(U, U_hat):
    """
    Computes Frobenius norm between two subspace projectors.
    """
    return np.linalg.norm(
        U @ U.T - U_hat @ U_hat.T,
        ord="fro"
    )


def frob_norm_subspaces(datasets, estimates, do_U = True, do_Uf = True, do_Uk = True):
    
    if len(datasets) != len(estimates):
        raise ValueError(
            f"Length of Datasets {len(datasets)} does not match Estimates {len(estimates)}"
        )
    
    norms = []

    for data, estimate in zip(datasets, estimates):

        K = data.metadata["K"]

        if do_U:
            # Overall shared U
            norms.append({
                "K": K,
                "matrix": "U",
                "frob_norm": subspace_frob_norm(
                    data.U,
                    estimate.U
                )
            })
            
        if do_Uf:
            # Family subspaces
            for m, (U, U_hat) in enumerate(zip(data.Uf, estimate.Uf)):
                norms.append({
                    "K": K,
                    "matrix": f"Uf{m}",
                    "frob_norm": subspace_frob_norm(
                        U,
                        U_hat
                    )
                })

        if do_Uk:
            # Individual subspaces
            for k, (U, U_hat) in enumerate(zip(data.Uk, estimate.Uk)):
                norms.append({
                    "K": K,
                    "matrix": f"Uk{k}",
                    "frob_norm": subspace_frob_norm(
                        U,
                        U_hat
                    )
                })

    return norms