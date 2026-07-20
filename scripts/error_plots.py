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
    generate_simulation_data(spec_control, "control")
    elapsed = perf_counter() - start
    print(f'Control Simulation completed in: {elapsed} seconds')
    return elapsed

def variation_K(base: SimulationSpec, start: int, n: int, step: int, name: str, track_time: bool = True):

    if start < 10:
        raise Warning(f'Start value of {start} is too small and may not form two distinct groups')

    simulations = []
    times = {}

    output_path = Path().cwd() / "data" / name
    k_values = np.arange(start, step * (n + 1), step)

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


def main():


    #control_simulation()

    # Variation in K

    datasets, times = variation_K(base, start = 25, n = 8, step = 25, name = "variation_in_K")

    estimates = []
    length = len(datasets)
    for i, data in enumerate(datasets):
        print(f"\nEstimating... ({i + 1} / {length}) \n")
        t0 = perf_counter()
        U_hat = estimate_data(data, data.metadata["r"], 
                              data.metadata["rfk"],
                              data.metadata["rk"])
        elapsed = perf_counter() - t0
        print(f'Estimation {i} with {data.metadata["K"]} completed in:\n{round(elapsed, 10)} seconds\n') 
    

    # Variation in n

    
    # Variation in r


    # Variation in rfk


    # Variation in rk


    # Variation in sigma


    # Variation in error




    return None


if __name__ == "__main__":
    main()