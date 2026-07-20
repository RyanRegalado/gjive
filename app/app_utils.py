from pathlib import Path

def find_simulations(data_dir: Path, exclude_dirs: set[str] = {"tests"}) -> list[str]:
    """
    Return simulation names while ignoring datasets inside excluded directories.
    """
    simulations = []

    for npz_file in data_dir.rglob("*.npz"):
        # Check if any parent directory is excluded
        if any(parent.name in exclude_dirs for parent in npz_file.parents):
            continue

        simulations.append(npz_file.parent.name)

    return sorted(set(simulations))