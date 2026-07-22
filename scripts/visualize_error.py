"""
Visualize Frobenius norm error across a parameter sweep.

CSV format:
    seed,parameter_name,parameter_value,matrix_name,frob_norm

Plots:
    U
    Uf_0
    Uf_1

Each point:
    y = median frob_norm across seeds
    error bar = standard deviation across seeds
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


MATRICES = [
    "U",
    "Uf_0",
    "Uf_1",
]


def summarize_errors(df):
    """
    Aggregate errors across seeds.

    Returns:
        matrix_name -> summary dataframe
    """

    summaries = {}

    for matrix in MATRICES:

        subset = df[df["matrix_name"] == matrix]

        summary = (
            subset
            .groupby("parameter_value")["frob_norm"]
            .agg(
                median="median",
                deviation="std",
            )
            .reset_index()
        )

        summaries[matrix] = summary

    return summaries


def plot_error(summaries, parameter_name):

    plt.figure(figsize=(8, 5))

    for matrix, summary in summaries.items():

        plt.errorbar(
            summary["parameter_value"],
            summary["median"],
            yerr=summary["deviation"],
            marker="o",
            capsize=4,
            label=matrix,
        )

    plt.xlabel(parameter_name)
    plt.ylabel("Frobenius Norm Error")

    plt.title(
        f"Subspace Error vs {parameter_name}"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.show()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv",
        help="Path to sweep CSV"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required = {
        "seed",
        "parameter_name",
        "parameter_value",
        "matrix_name",
        "frob_norm",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    parameter_name = df["parameter_name"].iloc[0]

    summaries = summarize_errors(df)

    plot_error(
        summaries,
        parameter_name
    )


if __name__ == "__main__":
    main()