import numpy as np
import pandas as pd
from time import perf_counter
from pathlib import Path
from typing import Any, Sequence
from dataclasses import replace
# GJIVE Functions
from gjive.generate import generate_simulation_data
from gjive.estimate import estimate_data
# Classes
from gjive.dataset import GjiveData
from gjive.estimate_class import GjiveEstimate
from gjive.simulation_spec import SimulationSpec
from gjive.estimate_spec import EstimateSpec
from experiment_result import ExperimentResult


def run_experiment(simulation_spec: SimulationSpec,
                  estimate_spec: EstimateSpec,
                  seed: int,
                  parameter_name: str,
                  parameter_value: Any,
                  do_U: bool = True,
                  do_Ufk: bool = True,
                  uk_mode: str = "none") -> list:
    
    UK_MODES = {"none", "all", "summary"}

    if uk_mode not in UK_MODES:
        raise ValueError(
            f"uk_mode must be one of {UK_MODES}"
        )
    
    spec = replace(
        simulation_spec, 
        **{parameter_name:parameter_value}
        )

    data = generate_simulation_data(
        spec,
        name = f"{parameter_name}_{parameter_value}"
        )
    
    est_spec = replace(
        estimate_spec,
        **{parameter_name:parameter_value}
    )
    
    estimate = estimate_data(data, est_spec)

    results = []

    if do_U:
        results.append(
            ExperimentResult(
                seed=seed,
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                subspace="U",
                frob_norm=subspace_error(
                    data.U,
                    estimate.U
                )
            )
        )

    if do_Ufk:
        for group in range(len(data.Uf)):
            results.append(
                ExperimentResult(
                    seed=seed,
                    parameter_name=parameter_name,
                    parameter_value=parameter_value,
                    subspace=f"Uf_{group}",
                    frob_norm=subspace_error(
                        data.Uf[group],
                        estimate.Uf[group]
                    )
                )
            )

    if uk_mode == "all":
        for k in range(len(data.Uk)):
            results.append(
                ExperimentResult(
                    seed=seed,
                    parameter_name=parameter_name,
                    parameter_value=parameter_value,
                    subspace=f"Uk_{k}",
                    frob_norm=subspace_error(
                        data.Uk[k],
                        estimate.Uk[k]
                    )
                )
            )

    elif uk_mode == "summary":
        uk_errors = []

        for k in range(len(data.Uk)):
            uk_errors.append(
                subspace_error(
                    data.Uk[k],
                    estimate.Uk[k]
                )
            )

        results.extend([
            ExperimentResult(
                seed=seed,
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                subspace="Uk_mean",
                frob_norm=float(np.mean(uk_errors))
            ),
            ExperimentResult(
                seed=seed,
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                subspace="Uk_std",
                frob_norm=float(np.std(uk_errors))
            ),
            ExperimentResult(
                seed=seed,
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                subspace="Uk_max",
                frob_norm=float(np.max(uk_errors))
            )
        ])
    
    return results

# def variation():



        
def subspace_error(U, U_hat):
    
    return np.linalg.norm(
        U @ U.T - U_hat @ U_hat.T, 
        ord = "fro"
        )

def validate_parameter(parameter_name, value):

    PARAMETER_TYPES = {
        "K": int,
        "n": int,
        "r": int,
        "snr": float,
        "rfk": list(int),
        "rk": list(int)
    }

    if value != PARAMETER_TYPES[parameter_name]:
        raise ValueError(
            f"Parameter name: {parameter_name} is not of type {str(PARAMETER_TYPES[parameter_name])}"
        )              




















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

def run_variation_trial(
    base: SimulationSpec,
    values: list[int],
    parameter: str,
    variation_name: str,
    seed: int,
):
    spec = replace(base, seed=seed)      # or build a new SimulationSpec

    datasets, _ = generate_variation(
        spec,
        values,
        parameter,
        f"{variation_name}/seed_{seed}",
    )

    estimates, _ = estimate_variation(
        datasets,
        parameter,
        f"{variation_name}/seed_{seed}",
    )

    return estimates




def get_datasets(name):
    datasets = []
    data_dir = Path().cwd() / "data" / name
    for file in data_dir.iterdir():
        datasets.append(GjiveData(file))
        print(f'Retrieved data file: {file}')
    return datasets

def get_estimates(name):
    estimates = []
    est_dir = Path().cwd() / "estimates" / name
    for file in est_dir.iterdir():
        estimates.append(GjiveEstimate(file))
        print(f'Retrieved estimate file: {file}')
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