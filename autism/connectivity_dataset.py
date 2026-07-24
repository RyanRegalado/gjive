import numpy as np
import pandas as pd
from pathlib import Path
import os

from sklearn.preprocessing import StandardScaler

DIR =  Path().cwd() / "autism" 

# =====================================================
# File Locations
# =====================================================

ROI_FOLDER = DIR / "abide_cc200_all"

PHENO_FILE = DIR / "csvs" / "Phenotypic_V1_0b_preprocessed1.csv"

OUTPUT_FILE = DIR / "csvs" / "processed_connectivity_dataset.csv"

# =====================================================
# Load phenotype data
# =====================================================

print("Loading phenotype file...")

pheno = pd.read_csv(PHENO_FILE)

pheno.columns = pheno.columns.str.strip()

pheno["SUB_ID"] = pd.to_numeric(
    pheno["SUB_ID"],
    errors="coerce"
)

pheno = pheno[["SUB_ID", "DX_GROUP"]]

print("Subjects in phenotype file:", len(pheno))

# =====================================================
# Upper triangle indices
# =====================================================

upper_idx = np.triu_indices(200, k=1)

feature_names = [
    f"{i+1}_{j+1}"
    for i, j in zip(*upper_idx)
]

print("Number of connectivity features:", len(feature_names))

# =====================================================
# Process every subject
# =====================================================

rows = []

files = sorted(os.listdir(ROI_FOLDER))

total = len(files)

print("Total ROI files:", total)

for count, filename in enumerate(files, start=1):

    if not filename.endswith(".1D"):
        continue

    try:

        file_path = os.path.join(ROI_FOLDER, filename)

        # -----------------------------------------
        # Read ROI time series
        # -----------------------------------------

        ts = pd.read_csv(
            file_path,
            sep=r"\s+",
            comment="#",
            header=None
        )

        # Sometimes there is an empty last column
        ts = ts.dropna(axis=1, how="all")

        if ts.shape[1] != 200:

            print(
                f"Skipping {filename} "
                f"(expected 200 ROIs, found {ts.shape[1]})"
            )
            continue

        # -----------------------------------------
        # Pearson correlation matrix
        # -----------------------------------------

        corr = ts.corr(method="pearson")

        corr = corr.to_numpy()

        # -----------------------------------------
        # Extract upper triangle
        # -----------------------------------------

        features = corr[upper_idx]

        # -----------------------------------------
        # Subject ID
        # -----------------------------------------

        # -----------------------------------------
        # Subject ID - Updated Logic
        # -----------------------------------------
        parts = filename.split("_")

        subject_id = None
       
        for part in parts:
            if part.isdigit():
                subject_id = int(part)
                break
       
        if subject_id is None:
            print(f"Skipping {filename}: Could not extract numeric Subject ID")
            continue
           
        row = [subject_id]

        row.extend(features.tolist())

        rows.append(row)

        if count % 25 == 0:

            print(
                f"Processed "
                f"{count}/{total}"
            )

    except Exception as e:

        print(filename)

        print(e)

# =====================================================
# Create dataframe
# =====================================================

columns = ["SUB_ID"]

columns.extend(feature_names)

connectivity_df = pd.DataFrame(
    rows,
    columns=columns
)

print()

print("Connectivity dataframe shape:")

print(connectivity_df.shape)

# =====================================================
# Merge with phenotype
# =====================================================

data = connectivity_df.merge(
    pheno,
    on="SUB_ID",
    how="inner"
)

print()

print("Merged dataframe shape:")

print(data.shape)

# =====================================================
# Remove missing diagnosis
# =====================================================

data = data.dropna(subset=["DX_GROUP"])

# =====================================================
# Separate X and y
# =====================================================

X = data[feature_names]

y = data["DX_GROUP"]

# =====================================================
# Standardize
# =====================================================

print()

print("Standardizing features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=feature_names
)

scaled_df.insert(
    0,
    "SUB_ID",
    data["SUB_ID"].values
)

scaled_df["DX_GROUP"] = y.values

# =====================================================
# Save
# =====================================================

scaled_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()

print("Finished!")

print("Saved to:")

print(OUTPUT_FILE)

print()

print("Final dataset shape:")

print(scaled_df.shape)