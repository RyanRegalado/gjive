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
from error_analysis.experiment_result import ExperimentResult
from error_analysis.paramater_sweep import ParameterSweep
from error_analysis.seed_sweep import SeedSweep


def run_experiment(simulation_spec: SimulationSpec,
                  estimate_spec: EstimateSpec,
                  seed: int,
                  parameter_name: str,
                  parameter_value: Any,
                  parent_dir: Path,
                  do_U: bool = True,
                  do_Ufk: bool = True,
                  uk_mode: str = "none") -> list:
    
    UK_MODES = {"none", "all", "summary"}

    name = f"{parameter_name}_{parameter_value}"

    if uk_mode not in UK_MODES:
        raise ValueError(
            f"uk_mode must be one of {UK_MODES}"
        )

    data_path = Path.cwd() / "data" / parent_dir
    
    data = generate_simulation_data(
        simulation_spec,
        simulation_name=name,
        output_path=data_path
        )

    estimate_path = Path.cwd() / "estimates" / parent_dir / name

    estimate = estimate_data(
        data,
        estimate_spec,
        output_path=estimate_path)

    results = []

    if do_U:
        results.append(
            ExperimentResult(
                seed=seed,
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                matrix_name="U",
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
                    matrix_name=f"Uf_{group}",
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
                    matrix_name=f"Uk_{k}",
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
                matrix_name="Uk_mean",
                frob_norm=float(np.mean(uk_errors))
            ),
            ExperimentResult(
                seed=seed,
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                matrix_name="Uk_std",
                frob_norm=float(np.std(uk_errors))
            ),
            ExperimentResult(
                seed=seed,
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                matrix_name="Uk_max",
                frob_norm=float(np.max(uk_errors))
            )
        ])
    
    return results

def run_parameter_sweep(base_spec: SimulationSpec, parameter_name: str, values: Sequence, seed: int, sweep_name: str | None = None):

    for value in values:
        validate_parameter(parameter_name, value)

    experiments = {}

    for value in values:

        print("Running value:", value)

        updates = {parameter_name: value}

        # Spec creation edge cases
        if parameter_name == "K":  
            updates['rk'] = [base_spec.rk[0]] * value
        
        new_base = replace(base_spec, **updates)

        new_base_est = EstimateSpec.from_simulation(new_base)
        # If you want Ufk data, specify it here. Come back later to create a struct or something to handle this.

        data_dir = Path(f'{sweep_name}') / f'seed_{seed}'
        experiment_results = run_experiment(
            new_base,
            new_base_est,
            seed,
            parameter_name,
            value,
            parent_dir = data_dir
        )


        experiments[value] = experiment_results

    return ParameterSweep(
        seed=seed,
        parameter_name=parameter_name,
        values=values,
        experiments=experiments
    )

def run_seed_sweep(
        base_spec: SimulationSpec,
        parameter_name: str,
        values: Sequence,
        seeds: Sequence[int],
        sweep_name: str | None = None,
    ) -> SeedSweep:

        sweeps = {}

        for seed in seeds:

            parameter_sweep = run_parameter_sweep(
                base_spec,
                parameter_name,
                values,
                seed,
                sweep_name,
            )

            sweeps[seed] = parameter_sweep

        return SeedSweep(
            parameter_name=parameter_name,
            sweeps=sweeps,
        )

        
def subspace_error(U, U_hat):
    
    return np.linalg.norm(
        U @ U.T - U_hat @ U_hat.T, 
        ord = "fro"
        )

def validate_parameter(parameter_name, value):
    valid_types = {
        "n": int,
        "K": int,
        "r": int,
        "rfk": list,
        "rk": list,
        "p": (int, float),
        "snr": (int, float),
    }

    if parameter_name not in valid_types:
        raise ValueError(f"Unknown parameter: {parameter_name}")

    expected_type = valid_types[parameter_name]

    if not isinstance(value, expected_type):
        raise ValueError(
            f"Parameter {parameter_name} value {value} is not of type {expected_type}"
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
    
def subspace_frob_norm(U, U_hat):
    """
    Computes Frobenius norm between two subspace projectors.
    """
    return np.linalg.norm(
        U @ U.T - U_hat @ U_hat.T,
        ord="fro"
    )

def save_sweep_results(
    sweep,
    output_path: Path,
):
    rows = []

    # -------------------------
    # SeedSweep
    # -------------------------
    if isinstance(sweep, SeedSweep):

        for seed, parameter_sweep in sweep.sweeps.items():

            for value, results in parameter_sweep.experiments.items():

                for result in results:
                    rows.append(
                        {
                            "seed": seed,
                            "parameter_name": parameter_sweep.parameter_name,
                            "parameter_value": value,
                            "matrix_name": result.matrix_name,
                            "frob_norm": result.frob_norm,
                        }
                    )


    # -------------------------
    # ParameterSweep
    # -------------------------
    elif isinstance(sweep, ParameterSweep):

        for value, results in sweep.experiments.items():

            for result in results:
                rows.append(
                    {
                        "seed": sweep.seed,
                        "parameter_name": sweep.parameter_name,
                        "parameter_value": value,
                        "matrix_name": result.matrix_name,
                        "frob_norm": result.frob_norm,
                    }
                )

    else:
        raise TypeError(
            "Expected ParameterSweep or SeedSweep"
        )


    df = pd.DataFrame(rows)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )