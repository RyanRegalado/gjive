from pathlib import Path
import pandas as pd


from gjive.dataset import GjiveData
from gjive.simulation_spec import SimulationSpec
from error_analysis.variation_utils import run_seed_sweep, sweep_results
from error_analysis.variation_utils import plot_parameter_sweep_df

def main():

    obj = [[i, i] for i in range(1, 21)]
    base_spec = SimulationSpec(
            n=100,
            K=50,
            r=3,
            rfk=[12, 12],
            rk=[8] * 50,
            p=0.5,
            snr=1,
            seed=1,
        )
    obj = [2,4,6,8,10,12,14,16,18,20]
        
    result = run_seed_sweep(base_spec, "r", obj, [1,2,3,4,5], "r_6_test")

    rows = sweep_results(result)

    df = pd.DataFrame(rows)

    plot_parameter_sweep_df(df, "r", show = True)

    return None





if __name__ == "__main__":
    main()