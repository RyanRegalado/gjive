"""
plot_sweep.py

Utility for plotting parameter sweep results produced by sweep_results().

Creates a single figure with median Frobenius norms and IQR error bars for
multiple matrices (e.g. U, Uf_0, Uf_1).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from error_analysis.variation_utils import plot_parameter_sweep

def main():
    params = [
        "n",
        "K",
        "r",
        "rfk",
        "rk",
        "p",
        "snr",
    ]
    for param in params:
        plot_parameter_sweep(
            csv_path= Path().cwd() / "error_analysis_results" / "csvs" / "full_sweep.csv", 
            parameter_name=param,
            matrices=["U", "Uf_0", "Uf_1"],
            save_path =  Path().cwd() / "error_analysis_results" / "figures" / f"{param}_plot"
        )

if __name__ == "__main__":
    main()
