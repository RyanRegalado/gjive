import os
import numpy as np
import pandas as pd
from pathlib import Path


# =====================================================
# File Locations
# =====================================================



CSV_DIR = Path.cwd() / "autism" / "csvs"

ROI_FOLDER = Path.cwd() / "autism" / "abide_cc200_all"

PHENO_FILE = CSV_DIR / "Phenotypic_V1_0b_preprocessed1.csv"

OUTPUT_FILE = CSV_DIR / "processed_connectivity_dataset.csv"


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
# Process subjects
# =====================================================

rows = []

files = sorted(os.listdir(ROI_FOLDER))

total = len(files)

print("Total ROI files:", total)


for count, filename in enumerate(files, start=1):

    if not filename.endswith(".1D"):
        continue

    try:

        file_path = os.path.join(
            ROI_FOLDER,
            filename
        )


        # -----------------------------------------
        # Read ROI time series
        # -----------------------------------------

        ts = pd.read_csv(
            file_path,
            sep=r"\s+",
            comment="#",
            header=None
        )


        ts = ts.dropna(axis=1, how="all")


        if ts.shape[1] != 200:

            print(
                f"Skipping {filename}: "
                f"{ts.shape[1]} columns"
            )

            continue


        # -----------------------------------------
        # Pearson correlation matrix
        # -----------------------------------------

        corr = ts.corr(
            method="pearson"
        )


        corr = corr.to_numpy()


        # Sanity check

        if corr.max() > 1.0001 or corr.min() < -1.0001:

            print("Invalid correlation detected:")
            print(filename)
            print(corr.max())
            print(corr.min())

            continue


        # -----------------------------------------
        # Extract unique connectivity features
        # -----------------------------------------

        features = corr[upper_idx]


        # -----------------------------------------
        # Extract subject ID
        # -----------------------------------------

        parts = filename.split("_")

        subject_id = None

        for part in parts:

            if part.isdigit():

                subject_id = int(part)
                break


        if subject_id is None:

            print(
                f"Skipping {filename}: "
                "No subject ID"
            )

            continue


        row = [subject_id]

        row.extend(
            features.tolist()
        )

        rows.append(row)


        if count % 25 == 0:

            print(
                f"Processed {count}/{total}"
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
print("Connectivity dataframe:")
print(connectivity_df.shape)



# =====================================================
# Merge phenotype
# =====================================================

data = connectivity_df.merge(
    pheno,
    on="SUB_ID",
    how="inner"
)


data = data.dropna(
    subset=["DX_GROUP"]
)


print()
print("Merged dataset:")
print(data.shape)



# =====================================================
# Save RAW Pearson connectivity values
# =====================================================

data.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("Saved raw connectivity dataset:")
print(OUTPUT_FILE)