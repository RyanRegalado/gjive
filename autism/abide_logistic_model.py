import numpy as np
import pandas as pd
from pathlib import Path 

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

DIR = Path().cwd() / "autism"

# =====================================================
# File Locations
# =====================================================

INPUT_FILE = DIR / "csvs" / "processed_connectivity_dataset.csv"

OUTPUT_MATRIX = DIR / "csvs" / "logistic_beta_matrix.csv"

OUTPUT_TABLE = DIR / "csvs" / "logistic_beta_table.csv"

# =====================================================
# Load processed dataset
# =====================================================

print("Loading processed dataset...")

data = pd.read_csv(INPUT_FILE)

print(data.shape)

# =====================================================
# Convert diagnosis to binary
#
# ABIDE:
# 1 = Autism
# 2 = Control
# =====================================================

data["DX_GROUP"] = (data["DX_GROUP"] == 1).astype(int)

# =====================================================
# Separate predictors and response
# =====================================================

X = data.drop(columns=["SUB_ID", "DX_GROUP"])

y = data["DX_GROUP"]

print()

print("Subjects:", len(y))
print("Features:", X.shape[1])

# =====================================================
# Drop subjects with high missing data
# =====================================================
# Threshold: drop if more than 10% of features are missing
threshold = 0.10 * X.shape[1]

# Create a boolean mask of valid subjects
valid_subjects_mask = X.isnull().sum(axis=1) <= threshold

# Apply the filter to both X and y
X = X[valid_subjects_mask]
y = y[valid_subjects_mask]

print(f"Dropped {sum(~valid_subjects_mask)} subjects due to excessive missing data.")
print(f"Remaining subjects: {len(y)}")

# =====================================================
# Fit Logistic Regression
# =====================================================

print()

print("Fitting Logistic Regression...")

# 1. Initialize and apply imputer to fill remaining NaNs with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

model = LogisticRegression(
    penalty="l2",
    solver="liblinear",
    max_iter=5000,
    random_state=42
)

model.fit(X_imputed, y)

print("Finished.")

# =====================================================
# Get coefficients
# =====================================================

beta = model.coef_[0]

print()

print("Recovered", len(beta), "coefficients.")

# =====================================================
# Save coefficient table
# =====================================================

coef_table = pd.DataFrame({

    "Feature": X.columns,

    "Coefficient": beta

})

coef_table.to_csv(

    OUTPUT_TABLE,

    index=False

)

print()

print("Saved coefficient table.")

# =====================================================
# Reconstruct 200 x 200 coefficient matrix
# =====================================================

print()

print("Reconstructing coefficient matrix...")

beta_matrix = np.zeros((200, 200))

for feature, value in zip(X.columns, beta):

    roi1, roi2 = feature.split("_")

    roi1 = int(roi1) - 1
    roi2 = int(roi2) - 1

    beta_matrix[roi1, roi2] = value
    beta_matrix[roi2, roi1] = value

np.fill_diagonal(beta_matrix, 0)

# =====================================================
# Save matrix
# =====================================================

beta_df = pd.DataFrame(beta_matrix)

beta_df.index = np.arange(1, 201)

beta_df.columns = np.arange(1, 201)

beta_df.to_csv(

    OUTPUT_MATRIX

)

print()

print("Coefficient matrix shape:")

print(beta_df.shape)

print()

print("Saved coefficient matrix to:")

print(OUTPUT_MATRIX)

print()

print("Done.")