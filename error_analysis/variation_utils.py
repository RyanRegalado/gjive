"""Utilities for parameter and seed sweeps in GJIVE experiments.

Common parameter meanings:
- base_spec: base SimulationSpec used to generate simulation data.
- parameter_name: the name of the parameter being varied.
- values: the candidate values for the sweep.
- seed: RNG seed for reproducibility.
- sweep_name: optional folder name used to organize sweep outputs.
- parallel: whether to execute independent runs concurrently.
- n_jobs: number of worker processes for joblib.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from gjive.estimate import estimate_data
from gjive.generate import generate_simulation_data
from gjive.estimate_spec import EstimateSpec
from gjive.simulation_spec import SimulationSpec
from error_analysis.experiment_result import ExperimentResult
from error_analysis.paramater_sweep import ParameterSweep
from error_analysis.seed_sweep import SeedSweep

VALID_PARAMETER_TYPES = {
    "n": int,
    "K": int,
    "r": int,
    "rfk": list,
    "rk": list,
    "p": (int, float),
    "snr": (int, float),
}


def run_experiment(
    simulation_spec: SimulationSpec,
    estimate_spec: EstimateSpec,
    seed: int,
    parameter_name: str,
    parameter_value: Any,
    parent_dir: Path,
    do_U: bool = True,
    do_Ufk: bool = True,
    uk_mode: str = "none",
) -> list[ExperimentResult]:
    
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

def run_parameter_value(
    base_spec: SimulationSpec,
    parameter_name: str,
    value: Any,
    seed: int,
    sweep_name: str | None = None,
) -> tuple[Any, list[ExperimentResult]]:
    """Configure one parameter value and run the experiment.

    Parameters
    ----------
    base_spec : SimulationSpec
        Base simulation configuration.
    parameter_name : str
        Name of the parameter being varied.
    value : Any
        Parameter value to test.
    seed : int
        Random seed used for reproducible generation.
    sweep_name : str | None, optional
        Optional folder name for output organization.

    Returns
    -------
    tuple[Any, list[ExperimentResult]]
        The parameter value and its experiment results.
    """

    updates = {
        parameter_name: value,
        "seed": seed,
    }

    if parameter_name == "K":
        updates["rk"] = [base_spec.rk[0]] * value

    new_base = replace(base_spec, **updates)

    new_base_est = EstimateSpec.from_simulation(new_base)

    data_dir = (
        Path(sweep_name) / f"seed_{seed}" / f"{parameter_name}_{value}"
        if sweep_name
        else None
    )

    results = run_experiment(
        new_base,
        new_base_est,
        seed,
        parameter_name,
        value,
        parent_dir=data_dir
    )

    return value, results

def run_parameter_sweep(
    base_spec: SimulationSpec,
    parameter_name: str,
    values: Sequence[Any],
    seed: int,
    sweep_name: str | None = None,
    parallel: bool = True,
    n_jobs: int = -1,
) -> ParameterSweep:

    """Run a parameter sweep over a sequence of values.

    Parameters
    ----------
    base_spec : SimulationSpec
        Base simulation configuration.
    parameter_name : str
        Parameter to vary.
    values : Sequence[Any]
        Values to evaluate in the sweep.
    seed : int
        Seed for reproducible generation.
    sweep_name : str | None, optional
        Optional root folder for outputs.
    parallel : bool, optional
        Whether to execute experiments in parallel.
    n_jobs : int, optional
        Number of worker processes for joblib.

    Returns
    -------
    ParameterSweep
        Sweep results keyed by value.
    """

    for value in values:
        validate_parameter(parameter_name, value)

    if parallel:

        output = Parallel(n_jobs=n_jobs)(
            delayed(run_parameter_value)(
                base_spec,
                parameter_name,
                value,
                seed,
                sweep_name
            )
            for value in values
        )

        experiments = dict(output)

    else:

        experiments = {}

        for value in values:
            key, result = run_parameter_value(
                base_spec,
                parameter_name,
                value,
                seed,
                sweep_name
            )

            experiments[key] = result


    return ParameterSweep(
        seed=seed,
        parameter_name=parameter_name,
        values=values,
        experiments=experiments
    )

def run_seed_sweep(
    base_spec: SimulationSpec,
    parameter_name: str,
    values: Sequence[int],
    seeds: Sequence[int],
    sweep_name: str,
    parallel: bool = True,
) -> SeedSweep:
    """Run multiple parameter sweeps across several seeds.

    Parameters
    ----------
    base_spec : SimulationSpec
        Base simulation configuration.
    parameter_name : str
        Parameter being varied.
    values : Sequence[int]
        Parameter values to test.
    seeds : Sequence[int]
        Random seeds to use for each sweep.
    sweep_name : str | None, optional
        Optional output root name.
    parallel : bool, optional
        Whether each parameter sweep is parallelized.

    Returns
    -------
    SeedSweep
        Results keyed by seed.
    """

    sweeps: dict[int, ParameterSweep] = {}
    for seed in seeds:
        print(f"Running sweep with seed: {seed}")

        parameter_sweep = run_parameter_sweep(
            base_spec,
            parameter_name,
            values,
            seed,
            sweep_name,
            parallel,
        )

        sweeps[seed] = parameter_sweep

    return SeedSweep(
        parameter_name=parameter_name,
        sweeps=sweeps,
    )

def subspace_error(U: np.ndarray, U_hat: np.ndarray) -> float:
    """Compute Frobenius norm of the difference between two subspace projectors."""
    return np.linalg.norm(U @ U.T - U_hat @ U_hat.T, ord="fro")

def validate_parameter(parameter_name: str, value: Any) -> None:
    """Validate the type of a sweep parameter value."""
    if parameter_name not in VALID_PARAMETER_TYPES:
        raise ValueError(f"Unknown parameter: {parameter_name}")

    expected_type = VALID_PARAMETER_TYPES[parameter_name]
    if not isinstance(value, expected_type):
        raise ValueError(
            f"Parameter {parameter_name} value {value} is not of type {expected_type}"
        )
    
def subspace_frob_norm(U, U_hat):
    """
    Computes Frobenius norm between two subspace projectors.
    """
    return np.linalg.norm(
        U @ U.T - U_hat @ U_hat.T,
        ord="fro"
    )

def sweep_results(
    sweep,
    save_to_disk: bool = False,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Convert sweep results to rows and optionally save to CSV."""
    if save_to_disk and output_path is None:
        raise ValueError("Save to Disk enabled: please provide an output path")

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

    if save_to_disk:
        df = pd.DataFrame(rows)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            output_path,
            index=False
    )
    return rows