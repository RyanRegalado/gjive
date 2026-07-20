import streamlit as st
from pathlib import Path

from gjive.specifications import SimulationSpec
from gjive.generate import generate_simulation_data
from app_utils import find_simulations

st.title("Data Generator 😎")

invalid_names = find_simulations(Path().cwd() / "data")

name = st.text_input(label="Dataset Name", value="simulation_000x")

# ==========================
# Dataset Dimensions
# ==========================

st.header("Dataset Dimensions")

K = st.number_input(
    "Number of Matrices (K)",
    min_value=1,
    value=20,
    step=1,
)

n = st.number_input(
    "Number of Observations (n)",
    min_value=1,
    value=100,
    step=1,
)

seed = st.number_input(
    "Random Seed",
    min_value=0,
    value=1,
    step=1,
)

# ==========================
# Signal Ranks
# ==========================

st.header("Signal Ranks")

r = st.number_input(
    "Joint Rank",
    min_value=0,
    value=3,
)

col1, col2 = st.columns(2)

with col1:
    rf0 = st.number_input(
        "Group 0 Rank",
        min_value=0,
        value=2,
    )

with col2:
    rf1 = st.number_input(
        "Group 1 Rank",
        min_value=0,
        value=2,
    )

# ==========================
# Individual Ranks
# ==========================

st.header("Individual Ranks")

same_rank = st.checkbox(
    "Use the same individual rank for every matrix",
    value=True,
)

if same_rank:

    common_rank = st.number_input(
        "Individual Rank",
        min_value=0,
        value=2,
    )

    rk = [common_rank] * K

else:

    rk = []

    cols = st.columns(4)

    for k in range(K):

        with cols[k % 4]:
            rk.append(
                st.number_input(
                    f"Matrix {k+1}",
                    min_value=0,
                    value=2,
                    key=f"rk_{k}",
                )
            )

# ==========================
# Advanced Options
# ==========================

with st.expander("Advanced Options"):

    customize_p = st.checkbox(
        "Customize group assignment probability: *p*",
        value=False,
    )

    customize_signal = st.checkbox(
        "Customize signal strength: *σ*",
        value=False,
    )

    customize_noise = st.checkbox(
        "Customize noise: *E*",
        value=False,
    )

    # Default values
    p = None
    signal_scale = None
    noise = None

    if customize_p:
        p = st.slider(
            "Probability of Group 1",
            min_value=0.01,
            max_value=0.99,
            value=0.50,
            step=0.01,
        )
    else:
        p = 0.5

    if customize_signal:
        signal_scale = st.slider(
            "Signal Strength (σ)",
            min_value=1.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

    if customize_noise:
        noise = st.slider(
            "Noise Level (E)",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
        )

# ==========================
# Generate
# ==========================

generate = st.button("Generate Dataset", type = "primary")

if generate:

    if name in set(invalid_names):
        raise FileExistsError(f'The simulation name: **{name}** already exists. Choose a different name.')

    try:

        spec = SimulationSpec(
            n=n,
            K=K,
            r=r,
            rfk=[rf0, rf1],
            rk=rk,
            p=p,
            seed=seed,
        )

        try:
            generate_simulation_data(spec, name)

            st.success("Successfully Generated Dataset!!")
            st.write(spec)

        except Exception as e:
            st.error(str(e))


    except ValueError as e:
        st.error(str(e))

    except FileExistsError as e:
        st.error(e)