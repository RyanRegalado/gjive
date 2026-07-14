# Global Dependencies

import numpy as np
from pathlib import Path
import streamlit as st

# Local Dependencies

from gjive.dataset import GjiveData
from app_utils import find_simulations


cwd = Path.cwd()

datasets = find_simulations(cwd / "data")

st.title("Dataset Overview")

choice = st.selectbox(
    label= "Choose a Dataset",
    options=datasets
)

data = GjiveData(cwd / "data" / choice)

st.write("### Specifications and Metadata")
st.table(data = data.metadata)