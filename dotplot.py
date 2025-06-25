import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load Excel data
df = pd.read_excel(
    "/home/xfang/Data/Multitasking_SMS/T1T2_mapvalues.xlsx", sheet_name="Sheet1"
)

df_run1 = df[(df["RunID"] == "run1") & (df["ReconMethod"].isin(["GMT", "MT"]))]

unique_subjects = {
    subj: f"Subject {i + 1}" for i, subj in enumerate(df_run1["Subject"].unique())
}
df_run1["Subject_Label"] = df_run1["Subject"].map(unique_subjects)

# Ensure MT is left, GMT is right
df_run1["ReconMethod"] = pd.Categorical(
    df_run1["ReconMethod"], categories=["MT", "GMT"], ordered=True
)

# Calculate group means
mean_values = df_run1.groupby("ReconMethod")["t1_cov"].mean()

# Plot
plt.figure(figsize=(8, 5))
sns.stripplot(
    data=df_run1,
    x="ReconMethod",
    y="t1_cov",
    hue="Subject_Label",
    jitter=True,
    size=8,
    palette="tab10",
)

# Add mean lines and annotations
for i, (method, mean_val) in enumerate(mean_values.items()):
    plt.hlines(y=mean_val, xmin=i - 0.2, xmax=i + 0.2, color="red", linestyle="--")
    plt.text(
        i + 0.05,
        mean_val + 0.005,
        f"Mean: {mean_val:.3f}",
        color="red",
        ha="center",
        fontsize=10,
    )

# Format plot
plt.title("T1 CoV Dot Plot: MT vs GMT")
plt.xlabel("Reconstruction Method")
plt.ylabel("T1 Coefficient of Variation")
plt.legend(bbox_to_anchor=(0.98, 0.98), loc="upper right")
plt.tight_layout()
plt.savefig("T1_CoV_DotPlot.eps", format="eps", bbox_inches="tight")

# Calculate group means
mean_values = df_run1.groupby("ReconMethod")["t2_cov"].mean()

# Plot
plt.figure(figsize=(8, 5))
sns.stripplot(
    data=df_run1,
    x="ReconMethod",
    y="t2_cov",
    hue="Subject_Label",
    jitter=True,
    size=8,
    palette="tab10",
)

# Add mean lines and annotations
for i, (method, mean_val) in enumerate(mean_values.items()):
    plt.hlines(y=mean_val, xmin=i - 0.2, xmax=i + 0.2, color="red", linestyle="--")
    plt.text(
        i + 0.05,
        mean_val + 0.005,
        f"Mean: {mean_val:.3f}",
        color="red",
        ha="center",
        fontsize=10,
    )

# Format plot
plt.title("T2 CoV Dot Plot: MT vs GMT")
plt.xlabel("Reconstruction Method")
plt.ylabel("T2 Coefficient of Variation")
plt.legend(bbox_to_anchor=(0.98, 0.98), loc="upper right")
plt.tight_layout()
plt.savefig("T2_CoV_DotPlot.eps", format="eps", bbox_inches="tight")
