import os
import torch

torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils
import torch.distributions
import torch.optim as optim
from tqdm import tqdm
import torchvision
import mat73
import numpy as np
import scipy
import matplotlib.pyplot as plt

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


# Load Excel file
# Load your data (replace with your actual path if needed)
df = pd.read_excel("/home/xfang/Data/Multitasking_SMS/T1T2_mapvalues_wo.xlsx", sheet_name="Sheet1")


df_run1 = df[df["RunID"] == "run1"]

# List of variables to test
metrics = ["T1_Mean_Septal", "T2_Mean_Septal", "t1_cov", "t2_cov"]

# Run ANOVAs on run1 data
for metric in metrics:
    print(f"\n--- ANOVA for {metric} (RunID = run1) ---")
    model = ols(f"{metric} ~ C(Subject) + C(Slice) + C(ReconMethod)", data=df_run1).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    

# Split the data by reconstruction method
df_gmt_run1 = df_run1[df_run1["ReconMethod"] == "GMT"]
df_mt_run1 = df_run1[df_run1["ReconMethod"] == "MT"]

mean_cov_gmt_t1_run1 = df_gmt_run1["t1_cov"].mean()
mean_cov_gmt_t2_run1 = df_gmt_run1["t2_cov"].mean()
mean_cov_mt_t1_run1 = df_mt_run1["t1_cov"].mean()
mean_cov_mt_t2_run1 = df_mt_run1["t2_cov"].mean()
mean_t2_mt_run1 = df_mt_run1["T2_Mean_Septal"].mean()
mean_t2_gmt_run1 = df_gmt_run1["T2_Mean_Septal"].mean()

# Print CoV results for run1
print("\n--- Average CoV Results (RunID = run1) ---")
print("Average CoV (GMT): T1 =", round(mean_cov_gmt_t1_run1, 3), ", T2 =", round(mean_cov_gmt_t2_run1, 3))
print("Average CoV (MT):  T1 =", round(mean_cov_mt_t1_run1, 3), ", T2 =", round(mean_cov_mt_t2_run1, 3))
print("Average T2 (GMT):   =", round(mean_t2_gmt_run1, 3))
print("Average T2 (MT):    =", round(mean_t2_mt_run1, 3))


# # Define a function to run ANOVA
# def run_anova(data, dependent_var):
#     model = ols(f"{dependent_var} ~ C(Subject) + C(Slice) + C(RunID)", data=data).fit()
#     anova_table = sm.stats.anova_lm(model, typ=2)
#     return anova_table

# # Run ANOVAs
# anova_gmt_t1 = run_anova(df_gmt, "T1_Mean_Septal")
# anova_mt_t1 = run_anova(df_mt, "T1_Mean_Septal")
# anova_gmt_t2 = run_anova(df_gmt, "T2_Mean_Septal")
# anova_mt_t2 = run_anova(df_mt, "T2_Mean_Septal")

# # Display results
# print("GMT - T1_Mean_Septal ANOVA")
# print(anova_gmt_t1)

# print("\nMT - T1_Mean_Septal ANOVA")
# print(anova_mt_t1)

# print("\nGMT - T2_Mean_Septal ANOVA")
# print(anova_gmt_t2)

# print("\nMT - T2_Mean_Septal ANOVA")
# print(anova_mt_t2)

# mean_cov_gmt_t1 = df_gmt["t1_cov"].mean()
# mean_cov_gmt_t2 = df_gmt["t2_cov"].mean()
# mean_cov_mt_t1 = df_mt["t1_cov"].mean()
# mean_cov_mt_t2 = df_mt["t2_cov"].mean()

# # --- Print CoV Results ---
# print("Average CoV (GMT): T1 =", round(mean_cov_gmt_t1, 3), ", T2 =", round(mean_cov_gmt_t2, 3))
# print("Average CoV (MT):  T1 =", round(mean_cov_mt_t1, 3), ", T2 =", round(mean_cov_mt_t2, 3))





# # --- T1_Mean_Septal ---
# model_t1 = ols('T1_Mean_Septal ~ C(Subject) + C(Slice) + C(RunID) + C(ReconMethod)', data=df).fit()
# anova_t1 = sm.stats.anova_lm(model_t1, typ=2)

# # --- T2_Mean_Septal ---
# model_t2 = ols('T2_Mean_Septal ~ C(Subject) + C(Slice) + C(RunID) + C(ReconMethod)', data=df).fit()
# anova_t2 = sm.stats.anova_lm(model_t2, typ=2)

# # --- t1_cov ---
# model_cov_t1 = ols('t1_cov ~ C(Subject) + C(Slice) + C(RunID) + C(ReconMethod)', data=df).fit()
# anova_cov_t1 = sm.stats.anova_lm(model_cov_t1, typ=2)

# # --- t2_cov ---
# model_cov_t2 = ols('t2_cov ~ C(Subject) + C(Slice) + C(RunID) + C(ReconMethod)', data=df).fit()
# anova_cov_t2 = sm.stats.anova_lm(model_cov_t2, typ=2)

# with pd.ExcelWriter("/home/xfang/Data/Multitasking_SMS/anova_results.xlsx", engine="openpyxl", mode='w') as writer:
#     # Write each table in a different section of Sheet2
#     anova_t1.to_excel(writer, sheet_name='Sheet2', startrow=0, startcol=0)
#     anova_t2.to_excel(writer, sheet_name='Sheet2', startrow=anova_t1.shape[0] + 3, startcol=0)
#     anova_cov_t1.to_excel(writer, sheet_name='Sheet2', startrow=0, startcol=6)
#     anova_cov_t2.to_excel(writer, sheet_name='Sheet2', startrow=anova_cov_t1.shape[0] + 3, startcol=6)


# test out repeatability

# df_runs = pd.read_excel('/home/xfang/Data/Multitasking_SMS/T1T2_mapvalues_wo.xlsx', sheet_name='Sheet2')
# print("Column names:", df.columns.tolist())
# # Perform ANOVA for T1_Mean_Septal across runs
# df_runs['Subject'] = df_runs['Subject'].astype(str)
# df_runs['Slice'] = df_runs['Slice'].astype(str)
# df_runs['ReconMethod'] = df_runs['ReconMethod'].astype(str)

# # Run ANOVA for T1_rep
# model_t1_run = ols('T1_rep ~ C(Subject) + C(Slice) + C(ReconMethod)', data=df_runs).fit()
# anova_t1_run = sm.stats.anova_lm(model_t1_run)

# # Run ANOVA for T2_rep
# model_t2_run = ols('T2_rep ~ C(Subject) + C(Slice) + C(ReconMethod)', data=df_runs).fit()
# anova_t2_run = sm.stats.anova_lm(model_t2_run)

# # Save the ANOVA tables to Excel
# output_path = "/home/xfang/Data/Multitasking_SMS/repeatability_anova_results.xlsx"
# with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
#     anova_t1_run.to_excel(writer, sheet_name='ANOVA_T1_rep')
#     anova_t2_run.to_excel(writer, sheet_name='ANOVA_T2_rep')