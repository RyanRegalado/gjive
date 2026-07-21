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

def estimate_variation(
    datasets: list[GjiveData],
    parameter: str,
    name: str,
    track_time: bool = True,
):
    estimates = []
    times = []

    output_path = Path.cwd() / "estimates" / name

    for i, data in enumerate(datasets):
        print(f"\nEstimating... ({i+1}/{len(datasets)})")

        if track_time:
            t0 = perf_counter()

        estimates.append(
            estimate_data(
                data,
                data.metadata["r"],
                data.metadata["rfk"],
                data.metadata["rk"],
                output_path / f'est_{data.metadata[parameter]}'
            )
        )

        if track_time:
            elapsed = perf_counter() - t0
            print(
                f'{parameter}={data.metadata[parameter]} '
                f'completed in {elapsed:.4f} seconds'
            )
            times.append(elapsed)

    return estimates, times

def generate_variation(
    base: SimulationSpec,
    values: list[int],
    parameter: str,
    name: str,
    track_time: bool = True,
):
    simulations = []
    times = {}

    output_path = Path.cwd() / "data" / name

    for i, value in enumerate(map(int, values)):

        kwargs = {
            "n": base.n,
            "K": base.K,
            "r": base.r,
            "rfk": base.rfk.copy(),
            "rk": base.rk.copy(),
            "p": base.p,
            "seed": base.seed,
            "signal_scale": base.signal_scale,
            "noise": base.noise,
        }

        if parameter == "K":
            kwargs["K"] = value
            kwargs["rk"] = [base.rk[0]] * value

        elif parameter == "n":
            kwargs["n"] = value

        elif parameter == "r":
            kwargs["r"] = value

        else:
            raise ValueError(f"Unsupported parameter '{parameter}'")

        current_spec = SimulationSpec(**kwargs)

        if track_time:
            t0 = perf_counter()

        sim = generate_simulation_data(
            current_spec,
            f"sim_{value}",
            output_path,
        )

        simulations.append(sim)

        if track_time:
            elapsed = perf_counter() - t0
            times[value] = elapsed
            print(
                f"Simulation {i+1}: {parameter}={value} completed in "
                f"{elapsed:.4f} seconds"
            )

    return simulations, times

def ticks(start: int, n: int, step: int):

    ticks = np.arange(start, step * (n + 1), step)

    return ticks

def get_datasets(name):
    datasets = []
    data_dir = Path().cwd() / "data" / name
    for file in data_dir.iterdir():
        datasets.append(GjiveData(file))
    return datasets

def get_estimates(name):
    estimates = []
    est_dir = Path().cwd() / "estimates" / name
    for file in est_dir.iterdir():
        estimates.append(GjiveEstimate(file))
    return estimates
    



def subspace_frob_norm(U, U_hat):
    """
    Computes Frobenius norm between two subspace projectors.
    """
    return np.linalg.norm(
        U @ U.T - U_hat @ U_hat.T,
        ord="fro"
    )


def frob_norm_subspaces(
    datasets,
    estimates,
    parameter,
    do_U=True,
    do_Uf=True,
    do_Uk=True,
):

    if len(datasets) != len(estimates):
        raise ValueError(
            f"Length of Datasets {len(datasets)} does not match Estimates {len(estimates)}"
        )

    norms = []

    for data, estimate in zip(datasets, estimates):

        value = data.metadata[parameter]

        if do_U:
            norms.append({
                parameter: value,
                "matrix": "U",
                "frob_norm": subspace_frob_norm(
                    data.U,
                    estimate.U,
                ),
            })

        if do_Uf:
            for m, (U, U_hat) in enumerate(zip(data.Uf, estimate.Uf)):
                norms.append({
                    parameter: value,
                    "matrix": f"Uf{m}",
                    "frob_norm": subspace_frob_norm(
                        U,
                        U_hat,
                    ),
                })

        if do_Uk:
            for k, (U, U_hat) in enumerate(zip(data.Uk, estimate.Uk)):
                norms.append({
                    parameter: value,
                    "matrix": f"Uk{k}",
                    "frob_norm": subspace_frob_norm(
                        U,
                        U_hat,
                    ),
                })

    return norms