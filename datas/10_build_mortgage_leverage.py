from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------
DATA_DIR = Path(".")

mortgage_file = DATA_DIR / "mortgage_lending_house_purchase_stocks_bsi_monthly.csv"
income_file = DATA_DIR / "gross_disposable_income_households_qsa_quarterly.csv"

# --------------------------------------------------
# 2. Countries used in the thesis
# --------------------------------------------------
countries = ["AT", "BE", "FI", "FR", "DE", "GR", "IE", "IT", "NL", "PT", "ES"]

# --------------------------------------------------
# 3. Load raw files
# --------------------------------------------------
mortgage_raw = pd.read_csv(mortgage_file)
income_raw = pd.read_csv(income_file)

# --------------------------------------------------
# 4. Clean mortgage lending stock
# --------------------------------------------------
mortgage = mortgage_raw.copy()

mortgage["date"] = pd.to_datetime(mortgage["DATE"])
mortgage["quarter"] = mortgage["date"].dt.to_period("Q")

# Keep end-of-quarter months only: March, June, September, December
mortgage = mortgage[mortgage["date"].dt.month.isin([3, 6, 9, 12])].copy()

# Identify country columns from ECB BSI codes
mortgage_cols = {}

for col in mortgage.columns:
    match = re.search(r"BSI\.M\.([A-Z]{2})\.", col)
    if match:
        country = match.group(1)
        if country in countries:
            mortgage_cols[col] = country

mortgage = mortgage[["quarter"] + list(mortgage_cols.keys())].copy()
mortgage = mortgage.rename(columns=mortgage_cols)

# Reshape to long format
mortgage = mortgage.melt(
    id_vars="quarter",
    var_name="country",
    value_name="mortgage_lending_stock"
)

# --------------------------------------------------
# 5. Clean gross disposable income
# --------------------------------------------------
income = income_raw.copy()

income["date"] = pd.to_datetime(income["DATE"])
income["quarter"] = income["date"].dt.to_period("Q")

# Identify country columns from ECB QSA codes
income_cols = {}

for col in income.columns:
    match = re.search(r"QSA\.Q\.N\.([A-Z]{2})\.", col)
    if match:
        country = match.group(1)
        if country in countries:
            income_cols[col] = country

income = income[["quarter"] + list(income_cols.keys())].copy()
income = income.rename(columns=income_cols)

# Reshape to long format
income = income.melt(
    id_vars="quarter",
    var_name="country",
    value_name="gross_disposable_income_quarterly"
)

# --------------------------------------------------
# 6. Construct annual disposable income
# --------------------------------------------------
# Mortgage lending is a stock.
# Disposable income is a quarterly flow.
# Therefore, we use the rolling sum of the last 4 quarters.
income = income.sort_values(["country", "quarter"]).reset_index(drop=True)

income["gross_disposable_income_annual"] = (
    income.groupby("country")["gross_disposable_income_quarterly"]
    .rolling(window=4, min_periods=4)
    .sum()
    .reset_index(level=0, drop=True)
)

# --------------------------------------------------
# 7. Merge numerator and denominator
# --------------------------------------------------
mortgage_leverage = mortgage.merge(
    income,
    on=["country", "quarter"],
    how="inner"
)

# --------------------------------------------------
# 8. Construct mortgage leverage ratio
# --------------------------------------------------
# Both variables are in millions.
# Multiplying by 100 expresses the ratio as a percentage.
mortgage_leverage["mortgage_leverage_ratio"] = (
    mortgage_leverage["mortgage_lending_stock"] /
    mortgage_leverage["gross_disposable_income_annual"]
) * 100

# --------------------------------------------------
# 9. Restrict sample
# --------------------------------------------------
start_q = pd.Period("2002Q1", freq="Q")
end_q = pd.Period("2025Q3", freq="Q")

mortgage_leverage = mortgage_leverage[
    (mortgage_leverage["quarter"] >= start_q) &
    (mortgage_leverage["quarter"] <= end_q)
].copy()

# Drop rows where annual income could not be computed
mortgage_leverage = mortgage_leverage.dropna(
    subset=["gross_disposable_income_annual", "mortgage_leverage_ratio"]
).copy()

# --------------------------------------------------
# 10. Sort and inspect
# --------------------------------------------------
mortgage_leverage = mortgage_leverage.sort_values(["country", "quarter"]).reset_index(drop=True)

print("\nMORTGAGE LEVERAGE - first 10 rows")
print(mortgage_leverage.head(10))

print("\nSHAPE")
print(mortgage_leverage.shape)

print("\nCOUNTRIES")
print(sorted(mortgage_leverage["country"].unique()))

print("\nQUARTER RANGE")
print(mortgage_leverage["quarter"].min(), "to", mortgage_leverage["quarter"].max())

print("\nROWS PER COUNTRY")
print(mortgage_leverage.groupby("country").size())

print("\nMISSING VALUES")
print(mortgage_leverage.isna().sum())

print("\nSUMMARY OF MORTGAGE LEVERAGE RATIO")
print(mortgage_leverage["mortgage_leverage_ratio"].describe())

# --------------------------------------------------
# 11. Save clean file
# --------------------------------------------------
mortgage_leverage_to_save = mortgage_leverage.copy()
mortgage_leverage_to_save["quarter"] = mortgage_leverage_to_save["quarter"].astype(str)

mortgage_leverage_to_save.to_csv("clean_mortgage_leverage.csv", index=False)

print("\nSaved file: clean_mortgage_leverage.csv")

# --------------------------------------------------
# 12. Plot mortgage leverage
# --------------------------------------------------
plot_df = mortgage_leverage.copy()
plot_df["quarter_date"] = plot_df["quarter"].dt.to_timestamp()

plt.figure(figsize=(12, 7))

for country in sorted(plot_df["country"].unique()):
    subset = plot_df[plot_df["country"] == country]
    plt.plot(
        subset["quarter_date"],
        subset["mortgage_leverage_ratio"],
        label=country
    )

plt.title("Mortgage debt to annual disposable income")
plt.xlabel("Quarter")
plt.ylabel("Mortgage leverage ratio (%)")
plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

plt.savefig("plot_mortgage_leverage.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved file: plot_mortgage_leverage.png")