from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# --------------------------------------------------
# 1. Load final panel
# --------------------------------------------------
DATA_DIR = Path(".")
panel_file = DATA_DIR / "final_panel_baseline.csv"

df = pd.read_csv(panel_file)

# --------------------------------------------------
# 2. Basic preparation
# --------------------------------------------------
# Make sure the panel is sorted correctly
df = df.sort_values(["country", "quarter"]).reset_index(drop=True)

# Convert quarter to string for fixed effects
df["quarter_str"] = df["quarter"].astype(str)

# --------------------------------------------------
# 3. Create dependent variable: quarterly house-price growth
# --------------------------------------------------
# Use log difference * 100 so coefficients are approximately percentage-point changes
df["dlog_house_price"] = df.groupby("country")["log_house_price"].diff() * 100

# --------------------------------------------------
# 4. Create lagged leverage
# --------------------------------------------------
# Lag leverage by one quarter within each country
df["leverage_lag"] = df.groupby("country")["leverage_ratio"].shift(1)

# Standardize lagged leverage for easier interpretation
mean_lev = df["leverage_lag"].mean()
std_lev = df["leverage_lag"].std()

df["leverage_lag_std"] = (df["leverage_lag"] - mean_lev) / std_lev

# --------------------------------------------------
# 5. Create interaction term
# --------------------------------------------------
df["shock_x_lev"] = df["ecb_shock"] * df["leverage_lag_std"]

# --------------------------------------------------
# 6. Keep estimation sample
# --------------------------------------------------
reg_df = df.dropna(subset=["dlog_house_price", "leverage_lag_std", "shock_x_lev"]).copy()

print("\nESTIMATION SAMPLE SHAPE")
print(reg_df.shape)

print("\nESTIMATION SAMPLE PERIOD")
print(reg_df["quarter"].min(), "to", reg_df["quarter"].max())

print("\nCOUNTRIES IN ESTIMATION SAMPLE")
print(sorted(reg_df["country"].unique()))

# --------------------------------------------------
# 7. Baseline regression
# --------------------------------------------------
# IMPORTANT:
# - country fixed effects: C(country)
# - quarter fixed effects: C(quarter_str)
# - no separate 'ecb_shock' main effect, because it is common to all countries
#   and would be absorbed by quarter fixed effects
#
# Coefficient of interest: shock_x_lev
# If positive ECB shock = tighter-than-expected ECB policy,
# then a negative coefficient on shock_x_lev would mean:
# tighter shocks reduce house-price growth more in highly leveraged countries.
# --------------------------------------------------

formula = """
dlog_house_price ~ leverage_lag_std + shock_x_lev + C(country) + C(quarter_str)
"""

model = smf.ols(formula=formula, data=reg_df)

# Cluster standard errors by country
# (with only 11 countries, interpret clustered inference cautiously)
results = model.fit(
    cov_type="cluster",
    cov_kwds={"groups": reg_df["country"]}
)

# --------------------------------------------------
# 8. Print results
# --------------------------------------------------
print("\nBASELINE REGRESSION RESULTS")
print(results.summary())

# --------------------------------------------------
# 9. Save a compact coefficient table
# --------------------------------------------------
coef_table = pd.DataFrame({
    "variable": results.params.index,
    "coefficient": results.params.values,
    "std_error": results.bse.values,
    "t_stat": results.tvalues.values,
    "p_value": results.pvalues.values
})

coef_table.to_csv("baseline_regression_coefficients.csv", index=False)

# Save summary as text
with open("baseline_regression_summary.txt", "w", encoding="utf-8") as f:
    f.write(results.summary().as_text())

print("\nSaved files:")
print("- baseline_regression_coefficients.csv")
print("- baseline_regression_summary.txt")

# --------------------------------------------------
# 10. Print coefficient of main interest clearly
# --------------------------------------------------
beta = results.params["shock_x_lev"]
pval = results.pvalues["shock_x_lev"]

print("\nMAIN COEFFICIENT OF INTEREST")
print(f"shock_x_lev coefficient = {beta:.6f}")
print(f"p-value = {pval:.6f}")

if beta < 0:
    print("\nInterpretation: tighter ECB shocks are associated with lower house-price growth")
    print("in more leveraged countries, which is consistent with your main hypothesis.")
elif beta > 0:
    print("\nInterpretation: tighter ECB shocks are associated with higher house-price growth")
    print("in more leveraged countries, which goes against your main hypothesis.")
else:
    print("\nInterpretation: the interaction effect is essentially zero in this baseline.")