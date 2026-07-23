from pathlib import Path
import numpy as np
import pandas as pd

from gjive.simulation_spec import SimulationSpec
from error_analysis.variation_utils import (
    run_seed_sweep,
    sweep_results
)

SEEDS = list(range(10))


def main():

    base_spec = SimulationSpec(
        n=200,
        K=100,
        r=3,
        rfk=[3, 3],
        rk=[3] * 100,
        p=0.5,
        snr=1,
        seed=1,
    )

    sweeps = {

        "K": list(range(25, 501, 25)),

        "r": list(range(1, 21)),

        "n": list(range(100, 501, 25)),

        "rfk": [[i, i] for i in range(1, 21)],

        "rk": [[i] * base_spec.K for i in range(1, 11)],

        "snr": [0.125, 0.25, 0.5, 1, 2, 4, 8, 16],

        "p": np.round(np.arange(0.1, 1.00, 0.1), 2).tolist(),
    }

    all_results = []

    for parameter, values in sweeps.items():

        print(f"\nRunning {parameter} sweep...")

        results = run_seed_sweep(
            base_spec=base_spec,
            parameter_name=parameter,
            values=values,
            sweep_name=f"full_sweep/variation_in_{parameter}",
            seeds=SEEDS,
        )

        all_results.extend(sweep_results(results))

    output_path = Path.cwd() / "error_analysis_results" / "csvs" / "full_sweep.csv"

    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)

    print()
    print("=" * 60)
    print(f"Saved {len(all_results)} simulations")
    print(output_path)
    print("=" * 60)

def rfk_test():

    obj = [[i, i] for i in range(1, 21)]
    base_spec = SimulationSpec(
            n=200,
            K=100,
            r=3,
            rfk=[3, 3],
            rk=[3] * 100,
            p=0.5,
            snr=1,
            seed=1,
        )
    
    result = run_seed_sweep(base_spec, "rfk", obj, [1,2], "rfk_test")

    print(f'{result.sweeps[1].experiments.keys()}')


    return None

def rk_test():

    base_spec = SimulationSpec(
            n=200,
            K=100,
            r=3,
            rfk=[3, 3],
            rk=[3] * 100,
            p=0.5,
            snr=1,
            seed=1,
        )

    obj = [[i] * base_spec.K for i in range(1, 11)]
    result = run_seed_sweep(base_spec, "rk", obj, [1,2], "rk_test")

    rows = sweep_results(result)

    df = pd.DataFrame(rows)
    path = Path().cwd() / "rktest.csv" 
    df.to_csv(path)

def p_test():
    base_spec = SimulationSpec(
        n=200,
        K=100,
        r=3,
        rfk=[3, 3],
        rk=[3] * 100,
        p=0.5,
        snr=1,
        seed=1,
    )

    obj = np.round(np.arange(0.1, 1.00, 0.1), 2).tolist()

    result = run_seed_sweep(base_spec, "p", obj, [1,2], "p_test")

if __name__ == "__main__":
    main()