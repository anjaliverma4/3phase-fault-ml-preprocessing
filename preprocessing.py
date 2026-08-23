import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/classData.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("1. LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset successfully loaded.")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. CHECK DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("2. DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ============================================================
# 3. DEFINE FEATURES AND TARGETS
# ============================================================

TARGETS = ["G", "C", "B", "A"]

FEATURES = [
    "Ia",
    "Ib",
    "Ic",
    "Va",
    "Vb",
    "Vc"
]

print("\nFeatures:")
print(FEATURES)

print("\nTarget variables:")
print(TARGETS)


# ============================================================
# 4. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("3. MISSING VALUE ANALYSIS")
print("=" * 70)

missing = df.isnull().sum()

print(missing)

print("\nTotal missing values:", missing.sum())

missing.to_csv(
    os.path.join(OUTPUT_DIR, "missing_values.csv")
)


# Missing value visualization

plt.figure(figsize=(10, 5))

missing.plot(kind="bar")

plt.title("Missing Values in Dataset")
plt.xlabel("Attributes")
plt.ylabel("Number of Missing Values")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "01_missing_values.png"),
    dpi=300
)

plt.show()


# ============================================================
# 5. DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("4. DUPLICATE ANALYSIS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

if duplicate_count > 0:

    df = df.drop_duplicates()

    print("Duplicate rows removed.")

else:

    print("No duplicate rows found.")


# ============================================================
# 6. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("5. DESCRIPTIVE STATISTICS")
print("=" * 70)

statistics = df[FEATURES].describe()

print(statistics)

statistics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "descriptive_statistics.csv"
    )
)


# ============================================================
# 7. OUTLIER DETECTION USING IQR
# ============================================================

print("\n" + "=" * 70)
print("6. OUTLIER ANALYSIS")
print("=" * 70)

Q1 = df[FEATURES].quantile(0.25)

Q3 = df[FEATURES].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

outlier_mask = (
    (df[FEATURES] < lower_bound) |
    (df[FEATURES] > upper_bound)
)

outlier_counts = outlier_mask.sum()

print("\nPotential outliers per feature:")

print(outlier_counts)

print(
    "\nTotal potential outlier values:",
    outlier_counts.sum()
)

outlier_counts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "outlier_counts.csv"
    )
)


# IMPORTANT:
# We DO NOT remove these outliers.
#
# In an electrical fault dataset, unusually high
# current/voltage values can represent actual fault
# conditions.


print("\nOutliers are RETAINED because they may")
print("represent genuine electrical fault conditions.")


# ============================================================
# 8. BOXPLOT BEFORE NORMALIZATION
# ============================================================

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df[FEATURES]
)

plt.title(
    "Electrical Features Before Normalization"
)

plt.xlabel("Electrical Features")

plt.ylabel("Measured Value")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_boxplot_before.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 9. FEATURE DISTRIBUTION
# ============================================================

df[FEATURES].hist(
    figsize=(14, 9),
    bins=40
)

plt.suptitle(
    "Distribution of Electrical Features",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_feature_distributions.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 10. CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("7. CORRELATION ANALYSIS")
print("=" * 70)

correlation = df[FEATURES].corr()

print(correlation)

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)

plt.title(
    "Correlation Matrix of Electrical Features"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_correlation_matrix.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 11. CREATE FAULT CODE
# ============================================================

print("\n" + "=" * 70)
print("8. FAULT CLASS ANALYSIS")
print("=" * 70)

# Create a single fault code from G,C,B,A
#
# Example:
#
# G C B A
# 0 0 0 0 -> No Fault
# 1 0 0 1 -> LG
# 0 1 1 0 -> LL
# 1 0 1 1 -> LLG
# 0 1 1 1 -> LLL
# 1 1 1 1 -> LLLG

df["Fault_Code"] = (
    df["G"].astype(str) +
    df["C"].astype(str) +
    df["B"].astype(str) +
    df["A"].astype(str)
)


def fault_name(code):

    fault_names = {

        "0000": "No Fault",

        "1001": "LG Fault",

        "0110": "LL Fault",

        "1011": "LLG Fault",

        "0111": "LLL Fault",

        "1111": "LLLG Fault"
    }

    return fault_names.get(
        code,
        "Other"
    )


df["Fault_Type"] = df["Fault_Code"].apply(
    fault_name
)


print("\nFault distribution:")

fault_distribution = (
    df["Fault_Type"]
    .value_counts()
)

print(fault_distribution)


fault_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "fault_distribution.csv"
    )
)


# ============================================================
# 12. FAULT DISTRIBUTION PLOT
# ============================================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Fault_Type",
    order=fault_distribution.index
)

plt.title(
    "Distribution of Electrical Fault Classes"
)

plt.xlabel("Fault Type")

plt.ylabel("Number of Samples")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_fault_distribution.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 13. PREPARE X AND Y
# ============================================================

X = df[FEATURES].copy()

y = df[TARGETS].copy()


# ============================================================
# 14. TRAIN-TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("9. TRAIN / TEST SPLIT")
print("=" * 70)

# Use Fault_Code for stratification.
#
# This keeps approximately the same fault distribution
# in training and testing datasets.

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.25,

    random_state=42,

    stratify=df["Fault_Code"]
)


print(
    "\nTotal samples:",
    len(X)
)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "\nTraining percentage:",
    round(
        len(X_train) / len(X) * 100,
        2
    ),
    "%"
)

print(
    "Testing percentage:",
    round(
        len(X_test) / len(X) * 100,
        2
    ),
    "%"
)


# ============================================================
# 15. MIN-MAX NORMALIZATION
# ============================================================

print("\n" + "=" * 70)
print("10. MIN-MAX NORMALIZATION")
print("=" * 70)


# Scale all features to [-1, +1]

scaler = MinMaxScaler(
    feature_range=(-1, 1)
)


# IMPORTANT:
# Fit scaler ONLY on training data.

X_train_scaled = scaler.fit_transform(
    X_train
)


# Apply same scaler to test data.

X_test_scaled = scaler.transform(
    X_test
)


# Convert arrays back into DataFrames

X_train_scaled = pd.DataFrame(

    X_train_scaled,

    columns=FEATURES,

    index=X_train.index
)


X_test_scaled = pd.DataFrame(

    X_test_scaled,

    columns=FEATURES,

    index=X_test.index
)


print("\nTraining data BEFORE normalization:")

print(
    X_train.head()
)


print("\nTraining data AFTER normalization:")

print(
    X_train_scaled.head()
)


print("\nMinimum values after normalization:")

print(
    X_train_scaled.min()
)


print("\nMaximum values after normalization:")

print(
    X_train_scaled.max()
)


# ============================================================
# 16. BEFORE vs AFTER STATISTICS
# ============================================================

before_stats = X_train.describe().T

after_stats = X_train_scaled.describe().T

comparison = pd.DataFrame({

    "Before_Min":
        before_stats["min"],

    "Before_Max":
        before_stats["max"],

    "After_Min":
        after_stats["min"],

    "After_Max":
        after_stats["max"]

})


print("\n" + "=" * 70)
print("BEFORE VS AFTER NORMALIZATION")
print("=" * 70)

print(comparison)


comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "normalization_comparison.csv"
    )
)


# ============================================================
# 17. BOXPLOT AFTER NORMALIZATION
# ============================================================

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=X_train_scaled
)

plt.title(
    "Electrical Features After Min-Max Normalization"
)

plt.xlabel(
    "Electrical Features"
)

plt.ylabel(
    "Normalized Value"
)

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_boxplot_after.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 18. SAVE PREPROCESSED TRAINING DATA
# ============================================================

train_processed = X_train_scaled.copy()

for target in TARGETS:

    train_processed[target] = y_train[target]


train_processed["Fault_Code"] = (
    train_processed["G"].astype(str) +
    train_processed["C"].astype(str) +
    train_processed["B"].astype(str) +
    train_processed["A"].astype(str)
)


train_processed["Fault_Type"] = (
    train_processed["Fault_Code"].apply(
        fault_name
    )
)


# ============================================================
# 19. SAVE PREPROCESSED TEST DATA
# ============================================================

test_processed = X_test_scaled.copy()

for target in TARGETS:

    test_processed[target] = y_test[target]


test_processed["Fault_Code"] = (
    test_processed["G"].astype(str) +
    test_processed["C"].astype(str) +
    test_processed["B"].astype(str) +
    test_processed["A"].astype(str)
)


test_processed["Fault_Type"] = (
    test_processed["Fault_Code"].apply(
        fault_name
    )
)


# ============================================================
# 20. SAVE CSV FILES
# ============================================================

train_processed.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "train_preprocessed.csv"
    ),

    index=False
)


test_processed.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "test_preprocessed.csv"
    ),

    index=False
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PREPROCESSING SUMMARY")
print("=" * 70)

print(
    "\nOriginal dataset:",
    "classData.csv"
)

print(
    "Original number of samples:",
    len(df)
)

print(
    "Number of features:",
    len(FEATURES)
)

print(
    "Features:",
    FEATURES
)

print(
    "Target variables:",
    TARGETS
)

print(
    "Missing values:",
    missing.sum()
)

print(
    "Duplicate rows:",
    duplicate_count
)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "Scaling method:",
    "Min-Max Scaling"
)

print(
    "Scaling range:",
    "[-1, +1]"
)


print("\nFault classes:")

for fault, count in fault_distribution.items():

    print(
        f"  {fault}: {count}"
    )


print("\nGenerated output files:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        " ->",
        filename
    )


print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)