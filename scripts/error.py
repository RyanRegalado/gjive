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
from gjive.specifications import SimulationSpec


base = SimulationSpec(
    n = 200,
    K = 100,
    r = 3,
    rfk = [3, 3],
    rk = [3] * 100,
    p = 0.5,
    seed = 1,
)

def control_simulation():
    start = perf_counter()
    generate_simulation_data(base, "control")
    elapsed = perf_counter() - start
    print(f'Control Simulation completed in: {elapsed} seconds')
    return elapsed


def variation_K():
    name = "variation_in_K"
    datasets = get_datasets(name)
    estimates = get_estimates(name)

    norms = frob_norm_subspaces(datasets, estimates, do_Uk=False)
    df = pd.DataFrame(norms)
    df.to_csv(Path().cwd() / "error_analysis" / "csvs" / "variation_in_K.csv")

def main():

    # Variation in K
    #variation_K()

    # Variation in n
    name = "variation_in_n"
    
    values = ticks(25, 20, 25)
    datasets, _ = generate_variation(base, values, "n", name)
    estimates, _ = estimate_variation(datasets, "n", name)

    #datasets = get_datasets(name)
    #estimates = get_estimates(name)
    
    norms = frob_norm_subspaces(datasets, estimates, "n", do_Uk=False)
    df = pd.DataFrame(norms)
    df.to_csv(Path().cwd() / "error_analysis" / "csvs" / "variation_in_n.csv")


    
    # Variation in r


    # Variation in rfk


    # Variation in rk


    # Variation in sigma


    # Variation in error


    return None


if __name__ == "__main__":
    main()