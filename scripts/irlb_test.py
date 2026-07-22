from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gjive.dataset import GjiveData
from gjive.estimate import estimate_data
from gjive.generate import generate_simulation_data
from gjive.simulation_spec import SimulationSpec


def generate_dataset(
    n: int,
    K: int,
    seed: int = 1,
    output_root: Path | None = None,
) -> GjiveData:
    """Create a small synthetic dataset for IRLB timing comparisons."""

    specs = SimulationSpec(
        n=n,
        K=K,
        r=2,
        rfk=[2, 2],
        rk=[2] * K,
        p=0.5,
        seed=seed,
    )

    dataset_name = f"irlb_speed_n{n}_k{K}_seed{seed}"
    output_path = output_root if output_root is not None else Path("data")
    return generate_simulation_data(specs, dataset_name, output_path=output_path)


def run_benchmark(values, fixed_n: int, fixed_k: int, variable_name: str, output_root: Path, plot_dir: Path) -> None:
    values_list = []
    irlb_true_times = []
    irlb_false_times = []

    print(f"\nSweeping {variable_name}")
    print("-" * 90)
    print(f"{'value':<8}{'irlb=True':<15}{'irlb=False':<15}")

    for value in values:
        if variable_name == "n":
            n = value
            K = fixed_k
        else:
            n = fixed_n
            K = value

        data = generate_dataset(n=n, K=K, seed=1 + value, output_root=output_root)

        start = time.perf_counter()
        estimate_data(
            data,
            r=data.metadata["r"],
            rfk=data.metadata["rfk"],
            rk=data.metadata["rk"],
            use_irlb=True,
            output_path=Path("estimates") / f"{data.metadata['dataset_name']}_irlb_true",
        )
        irlb_true_time = time.perf_counter() - start

        start = time.perf_counter()
        estimate_data(
            data,
            r=data.metadata["r"],
            rfk=data.metadata["rfk"],
            rk=data.metadata["rk"],
            use_irlb=False,
            output_path=Path("estimates") / f"{data.metadata['dataset_name']}_irlb_false",
        )
        irlb_false_time = time.perf_counter() - start

        values_list.append(value)
        irlb_true_times.append(irlb_true_time)
        irlb_false_times.append(irlb_false_time)

        print(f"{value:<8}{irlb_true_time:>10.3f}s{'':<5}{irlb_false_time:>10.3f}s")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(values_list, irlb_true_times, marker="o", label="irlb=True")
    ax.plot(values_list, irlb_false_times, marker="s", label="irlb=False")
    ax.set_xlabel(variable_name)
    ax.set_ylabel("time (s)")
    ax.set_title(f"IRLB timing vs. {variable_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"irlb_benchmark_{variable_name}.png", dpi=200)
    plt.close(fig)

    print(f"Saved plot to {plot_dir / f'irlb_benchmark_{variable_name}.png'}")


def benchmark_irlb() -> None:
    output_root = Path("data") / "irlb_benchmarks"
    output_root.mkdir(parents=True, exist_ok=True)

    plot_dir = Path().cwd()
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_values = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    k_values = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    run_benchmark(n_values, fixed_n=None, fixed_k=20, variable_name="n", output_root=output_root, plot_dir=plot_dir)
    run_benchmark(k_values, fixed_n=200, fixed_k=None, variable_name="k", output_root=output_root, plot_dir=plot_dir)


if __name__ == "__main__":
    benchmark_irlb()
