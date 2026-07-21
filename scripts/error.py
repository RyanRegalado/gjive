import numpy as np
import pandas as pd
from time import perf_counter
from pathlib import Path
# GJIVE Functions
from gjive.generate import generate_simulation_data
from gjive.estimate import estimate_data
# Error Analysis Functions
from error_analysis.variation_utils import estimate_variation, K_vec, frob_norm_subspaces, generate_variation
# Classes
from gjive.dataset import GjiveData
from gjive.estimate_class import GjiveEstimate
from gjive.specifications import SimulationSpec


base = SimulationSpec(
    n = 200,
    K = 1000,
    r = 3,
    rfk = [3, 3],
    rk = [3] * 1000,
    p = 0.5,
    seed = 1,
)

def control_simulation():
    start = perf_counter()
    generate_simulation_data(base, "control")
    elapsed = perf_counter() - start
    print(f'Control Simulation completed in: {elapsed} seconds')
    return elapsed


def main():

    # Variation in K
    name = "variation_in_K"
    vec = K_vec(25, 8, 25)
    datasets, _ = generate_variation(base, vec, "K", name)
    estimate_variation(datasets, "K", name)

    datasets = []
    data_dir = Path().cwd() / "data" / name
    for file in data_dir.iterdir():
        datasets.append(GjiveData(file))

    est_dir = Path().cwd() / "estimates" / name
    estimates = []
    for file in est_dir.iterdir():
        estimates.append(GjiveEstimate(file))

    norms = frob_norm_subspaces(datasets, estimates, do_Uk=False)
    df = pd.DataFrame(norms)
    df.to_csv(Path().cwd() / "error_analysis" / "csvs" / "variation_in_K.csv")

    # Variation in n

    
    # Variation in r


    # Variation in rfk


    # Variation in rk


    # Variation in sigma


    # Variation in error


    return None


if __name__ == "__main__":
    main()