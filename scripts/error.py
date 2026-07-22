import numpy as np
import pandas as pd
from time import perf_counter
from pathlib import Path
# GJIVE Functions
from gjive.generate import generate_simulation_data
from gjive.estimate import estimate_data
# Error Analysis Functions
from error_analysis.variation_utils import estimate_variation, generate_variation, ticks, frob_norm_subspaces, get_datasets, get_estimates
# Classes
from gjive.dataset import GjiveData
from gjive.estimate_class import GjiveEstimate
from gjive.simulation_spec import SimulationSpec


base = SimulationSpec(
    n = 200,
    K = 100,
    r = 3,
    rfk = [3, 3],
    rk = [3] * 100,
    p = 0.5,
    seed = 1,
    noise = 1
)

def control_simulation():
    start = perf_counter()
    generate_simulation_data(base, "control")
    elapsed = perf_counter() - start
    print(f'Control Simulation completed in: {elapsed} seconds')
    return elapsed


def run_variation(parameter: str, values):
    name = f"variation_in_{parameter}"

    datasets, _ = generate_variation(base, values, parameter, name)
    estimates, _ = estimate_variation(datasets, parameter, name)

    norms = frob_norm_subspaces(
        datasets,
        estimates,
        parameter,
        do_Uk=False,
    )

    df = pd.DataFrame(norms)
    df.to_csv(
        Path.cwd() / "error_analysis" / "csvs" / f"{name}.csv",
        index=False,
    )

def main():

    # Variation in K
    

    # Variation in n

    datasets = get_datasets("variation_in_r")

    estimates, _ = estimate_variation(datasets, "r", "variation_in_r_irlb")
    parameter = "r"

    norms = frob_norm_subspaces(
        datasets,
        estimates,
        parameter,
        do_Uk=False,
    )

    df = pd.DataFrame(norms)
    df.to_csv(
        Path.cwd() / "error_analysis" / "csvs" / "variation_in_r_irlb.csv",
        index=True,
    )






    # Variation in r
    

    # Variation in rfk


    # Variation in rk


    # Variation in sigma


    # Variation in error


    return None


if __name__ == "__main__":
    main()