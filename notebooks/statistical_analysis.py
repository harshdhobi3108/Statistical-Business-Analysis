"""
Statistical Business Analysis
Adjusted for actual dataset structure
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, shapiro
import statsmodels.api as sm


# ==========================================================
# 1. PATH SETUP
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")

os.makedirs(VISUALS_DIR, exist_ok=True)

sales_path = os.path.join(DATA_DIR, "sales_data.csv")
churn_path = os.path.join(DATA_DIR, "customer_churn.csv")


# ==========================================================
# 2. LOAD DATA
# ==========================================================

sales_df = pd.read_csv(sales_path)
churn_df = pd.read_csv(churn_path)

sales_df.columns = sales_df.columns.str.strip().str.lower()
churn_df.columns = churn_df.columns.str.strip().str.lower()

print("\nData Loaded Successfully")
print("Sales Data Shape:", sales_df.shape)
print("Churn Data Shape:", churn_df.shape)


# ==========================================================
# 3. DESCRIPTIVE STATISTICS (Total Sales)
# ==========================================================

print("\n================ DESCRIPTIVE STATISTICS ================")

mean = sales_df["total_sales"].mean()
median = sales_df["total_sales"].median()
mode = sales_df["total_sales"].mode()[0]
std = sales_df["total_sales"].std()
variance = sales_df["total_sales"].var()

print(f"Mean: {round(mean,2)}")
print(f"Median: {round(median,2)}")
print(f"Mode: {round(mode,2)}")
print(f"Standard Deviation: {round(std,2)}")
print(f"Variance: {round(variance,2)}")


# ==========================================================
# 4. DISTRIBUTION ANALYSIS
# ==========================================================

print("\n================ DISTRIBUTION ANALYSIS ================")

plt.figure()
sns.histplot(sales_df["total_sales"], kde=True)
plt.title("Total Sales Distribution")
plt.savefig(os.path.join(VISUALS_DIR, "histogram_sales.png"))
plt.close()

stat, p_value = shapiro(sales_df["total_sales"])
print("Shapiro-Wilk p-value:", round(p_value,4))


# ==========================================================
# 5. CORRELATION ANALYSIS
# ==========================================================

print("\n================ CORRELATION ANALYSIS ================")

correlation_matrix = sales_df[["total_sales", "quantity", "price"]].corr()
print(correlation_matrix)

plt.figure()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(VISUALS_DIR, "correlation_heatmap.png"))
plt.close()


# ==========================================================
# 6. HYPOTHESIS TESTING
# ==========================================================

print("\n================ HYPOTHESIS TESTING ================")

# Test 1: High vs Low Quantity Impact on Sales
median_quantity = sales_df["quantity"].median()

high_q = sales_df[sales_df["quantity"] > median_quantity]["total_sales"]
low_q = sales_df[sales_df["quantity"] <= median_quantity]["total_sales"]

stat1, p1 = ttest_ind(high_q, low_q)

print("\nTest 1: Quantity Impact on Sales")
print("p-value:", round(p1,4))


# Test 2: Regional Sales Difference (ANOVA)
groups = [group["total_sales"].values for name, group in sales_df.groupby("region")]
stat2, p2 = f_oneway(*groups)

print("\nTest 2: Regional Sales Difference")
print("p-value:", round(p2,4))


# Test 3: Churn vs Monthly Charges
stat3, p3 = ttest_ind(
    churn_df[churn_df["churn"] == "Yes"]["monthlycharges"],
    churn_df[churn_df["churn"] == "No"]["monthlycharges"]
)

print("\nTest 3: Churn vs Monthly Charges")
print("p-value:", round(p3,4))


# ==========================================================
# 7. CONFIDENCE INTERVAL (95%)
# ==========================================================

print("\n================ CONFIDENCE INTERVAL ================")

n = len(sales_df)
confidence = 0.95
t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
margin_error = t_critical * (std / np.sqrt(n))

lower = mean - margin_error
upper = mean + margin_error

print(f"Average Total Sales: {round(mean,2)} ± {round(margin_error,2)} (95% CI)")
print(f"Confidence Interval: {round(lower,2)} to {round(upper,2)}")


# ==========================================================
# 8. REGRESSION ANALYSIS
# ==========================================================

print("\n================ REGRESSION ANALYSIS ================")

X = sales_df[["quantity", "price"]]
y = sales_df["total_sales"]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())

plt.figure()
plt.scatter(sales_df["quantity"], sales_df["total_sales"])
plt.title("Quantity vs Total Sales")
plt.xlabel("Quantity")
plt.ylabel("Total Sales")
plt.savefig(os.path.join(VISUALS_DIR, "regression_plot.png"))
plt.close()

print("\nAnalysis Completed Successfully.")