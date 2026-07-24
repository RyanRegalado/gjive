import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path
from autism.utils import get_elbows

st.set_page_config(page_title="Matrix Debugger", layout="wide")
st.title(" Connectivity Matrix Per Subject")

# 1. Load data without any cleaning
@st.cache_data
def load_data():
    data_path = Path().cwd() / "autism" / "csvs" / "all_subjects_connectivity_long.csv"
    return pd.read_csv(data_path)

df = load_data()

# 2. Sidebar selection
subject_ids = sorted(df['Subject_ID'].unique())
selected_id = st.sidebar.selectbox("Select a Subject ID", subject_ids)

# 3. Process and display
st.subheader(f"Raw Data for Subject: {selected_id}")

subject_data = df[df['Subject_ID'] == selected_id]
matrix = subject_data.pivot(index='Region_A', columns='Region_B', values='Correlation')

# Display dimensions
st.write(f"### Matrix Dimensions: {matrix.shape[0]} rows x {matrix.shape[1]} columns")

# Display raw dataframe
st.dataframe(matrix, use_container_width=True)

# 4. Optional: Download button
csv = matrix.to_csv().encode('utf-8')
st.download_button(
    label="Download this raw matrix",
    data=csv,
    file_name=f'raw_matrix_{selected_id}.csv',
    mime='text/csv',
)


# --- 1. Histogram (Excluding Self-Correlations) ---
st.subheader("Distribution of Correlation Strengths")

corr_values = matrix.values.copy()

# Remove diagonal (all 1's)
corr_values = corr_values[
    ~np.eye(corr_values.shape[0], dtype=bool)
]

fig, ax = plt.subplots(figsize=(8, 4))

ax.hist(
    corr_values.flatten(),
    bins=50,
    color='skyblue',
    edgecolor='black'
)

ax.axvline(
    0,
    color='red',
    linestyle='--'
)

ax.set_xlabel("Correlation")
ax.set_ylabel("Frequency")
ax.set_title(
    "Correlation Distribution (Diagonal Removed)"
)

st.pyplot(fig)

st.write(
    f"Mean Correlation: {corr_values.mean():.4f}"
)

st.write(
    f"Median Correlation: {np.median(corr_values):.4f}"
)

st.write(
    f"Std Dev: {corr_values.std():.4f}"
)

# --- 2. Centrality (Using NetworkX) ---
# Thresholding to ignore weak noise
thresholded_matrix = matrix.mask(matrix < 0.3, 0)
G = nx.from_pandas_adjacency(thresholded_matrix)

st.subheader("Network Centrality")
# Calculate all three measures
degree = nx.degree_centrality(G)
#betweenness = nx.betweenness_centrality(G)
eigenvector = nx.eigenvector_centrality(G, max_iter=1000)

# Combine into one DataFrame
centrality_df = pd.DataFrame({
    'Degree': degree,
    #'Betweenness': betweenness,
    'Eigenvector': eigenvector
})

# Display table sorted by Degree
st.write("Centrality Measures for Regions:")
st.dataframe(centrality_df.sort_values(by='Degree', ascending=False), use_container_width=True)


# --- 3. Scree Plot (Eigenvalues) ---
st.subheader("Scree Plot")

eigenvalues = np.linalg.eigvals(matrix.fillna(0))

sorted_eig = np.sort(np.abs(eigenvalues))[::-1]

elbows = get_elbows(sorted_eig, n=3)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Subject Effective Rank",
        elbows[0]
    )

with col2:
    st.metric(
        "Secondary Elbow",
        elbows[1] if len(elbows) > 1 else "N/A"
    )

fig_scree, ax_scree = plt.subplots(figsize=(10, 5))

ax_scree.plot(
    range(1, len(sorted_eig) + 1),
    sorted_eig,
    "o-"
)

for elbow in elbows:

    ax_scree.axvline(
        elbow,
        color="red",
        linestyle="--",
        alpha=0.7
    )

    ax_scree.scatter(
        elbow,
        sorted_eig[elbow - 1],
        s=100,
        color="red"
    )

ax_scree.set_xlabel("Component")
ax_scree.set_ylabel("Eigenvalue")
ax_scree.set_title(
    "Eigenvalue Decay with Estimated Elbows"
)

st.pyplot(fig_scree)

st.write(
    f"Estimated Scree Elbows: {elbows}"
)

# --- 4. Box Plot of Singular Values across all subjects ---
st.subheader("Distribution of Singular Values (Across All Subjects)")

all_sv_data = []

for subj in subject_ids:
    subj_data = df[df['Subject_ID'] == subj]
    mat = subj_data.pivot(index='Region_A', columns='Region_B', values='Correlation').fillna(0)
    _, s, _ = np.linalg.svd(mat.values)
    all_sv_data.append(s)

sv_df = pd.DataFrame(all_sv_data, columns=[f'{i+1}' for i in range(200)])

if sv_df.sum().sum() > 0:
    median_sv = sv_df.median(axis=0).values

    # Get elbows and slice only the first 3
    sv_elbows = get_elbows(median_sv, n=5)[:3]

    st.metric("Population Effective Rank (Primary)", sv_elbows[0])
    st.write(f"Population Elbows (First 3): {sv_elbows}")

    fig_box, ax_box = plt.subplots(figsize=(14, 6))
    ax_box.boxplot(sv_df.values, positions=np.arange(1, 201))
   
    # Plot only the first 3 red lines
    for elbow in sv_elbows:
        ax_box.axvline(elbow, color='red', linestyle='--', linewidth=2)

    ax_box.set_xlabel("Singular Value Index")
    ax_box.set_ylabel("Singular Value Magnitude")
    ax_box.set_title("Distribution of Singular Values Across Subjects")
    st.pyplot(fig_box)

    # Display rank data for the 3 elbows
    elbow_df = pd.DataFrame({
        "Elbow Number": [1, 2, 3],
        "Estimated Rank": sv_elbows
    })
    st.write("Estimated Population Rank Locations")
    st.dataframe(elbow_df, use_container_width=True)