from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------
DATA_DIR = Path(".")

baseline_panel_file = DATA_DIR / "final_panel_baseline.csv"
mortgage_leverage_file = DATA_DIR / "clean_mortgage_leverage.csv"

# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------
panel = pd.read_csv(baseline_panel_file)
mortgage_leverage = pd.read_csv(mortgage_leverage_file)

# --------------------------------------------------
# 3. Prepare quarter variable
# --------------------------------------------------
panel["quarter"] = pd.PeriodIndex(panel["quarter"], freq="Q")
mortgage_leverage["quarter"] = pd.PeriodIndex(mortgage_leverage["quarter"], freq="Q")

# --------------------------------------------------
# 4. Keep only needed mortgage leverage columns
# --------------------------------------------------
mortgage_leverage = mortgage_leverage[
    ["country", "quarter", "mortgage_lending_stock",
     "gross_disposable_income_annual", "mortgage_leverage_ratio"]
].copy()

# --------------------------------------------------
# 5. Merge with baseline panel
# --------------------------------------------------
panel_mortgage = panel.merge(
    mortgage_leverage,
    on=["country", "quarter"],
    how="inner"
)

# --------------------------------------------------
# 6. Sort and inspect
# --------------------------------------------------
panel_mortgage = panel_mortgage.sort_values(["country", "quarter"]).reset_index(drop=True)

print("\nFINAL PANEL WITH MORTGAGE LEVERAGE - first 10 rows")
print(panel_mortgage.head(10))

print("\nSHAPE")
print(panel_mortgage.shape)

print("\nCOUNTRIES")
print(sorted(panel_mortgage["country"].unique()))

print("\nQUARTER RANGE")
print(panel_mortgage["quarter"].min(), "to", panel_mortgage["quarter"].max())

print("\nROWS PER COUNTRY")
print(panel_mortgage.groupby("country").size())

print("\nMISSING VALUES")
print(panel_mortgage.isna().sum())

# --------------------------------------------------
# 7. Save final robustness panel
# --------------------------------------------------
panel_to_save = panel_mortgage.copy()
panel_to_save["quarter"] = panel_to_save["quarter"].astype(str)

panel_to_save.to_csv("final_panel_mortgage_leverage.csv", index=False)

print("\nSaved file: final_panel_mortgage_leverage.csv")