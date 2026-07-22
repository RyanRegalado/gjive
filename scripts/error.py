import numpy as np
import pandas as pd
from time import perf_counter
from pathlib import Path
# GJIVE Functions
from gjive.generate import generate_simulation_data
from gjive.estimate import estimate_data
# Error Analysis Functions
from error_analysis.variation_utils import estimate_variation, generate_variation, save_sweep_results
# Classes
from gjive.dataset import GjiveData
from gjive.estimate_class import GjiveEstimate
from gjive.simulation_spec import SimulationSpec
from gjive.estimate_spec import EstimateSpec

from error_analysis.variation_utils import run_experiment, run_parameter_sweep, run_seed_sweep
from error_analysis.experiment_result import ExperimentResult


base = SimulationSpec(
    n = 200,
    K = 100,
    r = 3,
    rfk = [3, 3],
    rk = [3] * 100,
    p = 0.5,
    seed = 1,
    snr = 0.2
)

def main():
    par = run_parameter_sweep(base, "K", [20, 30, 40, 50, 60, 70], 1, "test_par_sweep")

    print(f"Parameter sweep: {par.parameter_name}")
    print(f"Number of parameter values tested: {len(par.experiments)}\n")

    for parameter_value, experiments in par.experiments.items():
        print(f"{par.parameter_name} = {parameter_value}")

        for result in experiments:
            print(
                f"  {result.matrix_name}: "
                f"Frobenius error = {result.frob_norm:.4f}"
            )

        print()

def seed_test():
    sweep = run_seed_sweep(
            base,
            "K",
            [20, 30, 40, 50],
            [1, 2, 3],
            "test_seed_sweep"
        )

    print(f"Parameter: {sweep.parameter_name}")
    print(f"Seeds tested: {list(sweep.sweeps.keys())}")

    for seed, parameter_sweep in sweep.sweeps.items():
        print(f"\nSeed = {seed}")

        for value, results in parameter_sweep.experiments.items():
            print(f"  K = {value}: {len(results)} results")

            for result in results:
                print(
                    f"    {result.matrix_name}: "
                    f"{result.frob_norm:.4f}"
                )
    return None

def test_save_seed_sweep():

    sweep = run_seed_sweep(
        base,
        parameter_name="K",
        values=[20, 30],
        seeds=[1, 2],
        sweep_name="test_seed_sweep",
    )

    output_file = Path().cwd() / "error_analysis" / "csvs" / "test_seed_sweep.csv"

    save_sweep_results(
        sweep,
        output_file,
    )

    print(f"Saved results to: {output_file}")

    # Verify file contents
    df = pd.read_csv(output_file)

    print("\nCSV Preview:")
    print(df.head())

    print("\nRows:", len(df))

def test_K_sweep():

    K_values = list(range(10, 101, 10))
    seeds = list(range(1, 8))

    print(f"K values: {K_values}")
    print(f"Seeds: {seeds}")

    sweep = run_seed_sweep(
        base,
        parameter_name="K",
        values=K_values,
        seeds=seeds,
        sweep_name="K_sweep",
    )

    output_path = Path("results") / "K_sweep.csv"

    save_sweep_results(
        sweep,
        output_path,
    )

    print(f"\nSaved results to: {output_path}")

    # Optional sanity check
    import pandas as pd

    df = pd.read_csv(output_path)

    print("\nResults summary:")
    print(df.head())

    print("\nRows generated:", len(df))

    expected_rows = (
        len(K_values)
        * len(seeds)
        * 3   # U, Uf_0, Uf_1
    )

    print("Expected rows:", expected_rows)

    assert len(df) == expected_rows


if __name__ == "__main__":
    test_K_sweep()