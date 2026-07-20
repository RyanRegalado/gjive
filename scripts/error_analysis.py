import numpy as np
import pandas as pd
from time import perf_counter
from pathlib import Path
# GJIVE Functions
from gjive.generate import generate_simulation_data
from gjive.estimate import estimate_data
# Error Analysis Functions
from error_analysis.variation import generate_variation_K, estimate_variation_K, K_vec, frob_norm_subspaces
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

    #control_simulation()

    # Variation in K
    k_values = K_vec(25, 8, 25)

    datasets, times = generate_variation_K(base, k_values, name = "variation_in_K")

    estimates, times = estimate_variation_K(datasets)

    


    # Variation in n

    
    # Variation in r


    # Variation in rfk


    # Variation in rk


    # Variation in sigma


    # Variation in error


    return None


if __name__ == "__main__":
    main()