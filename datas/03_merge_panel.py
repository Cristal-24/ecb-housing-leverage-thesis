from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------
DATA_DIR = Path(".")

house_prices_file = DATA_DIR / "clean_house_prices.csv"
rents_file = DATA_DIR / "clean_rents.csv"
leverage_file = DATA_DIR / "clean_leverage.csv"
shocks_file = DATA_DIR / "clean_ecb_shocks.csv"

# --------------------------------------------------
# 2. Load cleaned datasets
# --------------------------------------------------
house_prices = pd.read_csv(house_prices_file)
rents = pd.read_csv(rents_file)
leverage = pd.read_csv(leverage_file)
shocks = pd.read_csv(shocks_file)

# --------------------------------------------------
# 3. Convert quarter columns to quarterly periods
# --------------------------------------------------
house_prices["quarter"] = pd.PeriodIndex(house_prices["quarter"], freq="Q")
rents["quarter"] = pd.PeriodIndex(rents["quarter"], freq="Q")
leverage["quarter"] = pd.PeriodIndex(leverage["quarter"], freq="Q")
shocks["quarter"] = pd.PeriodIndex(shocks["quarter"], freq="Q")

# --------------------------------------------------
# 4. Restrict all datasets to the agreed sample
# --------------------------------------------------
start_q = pd.Period("2002Q1", freq="Q")
end_q = pd.Period("2025Q3", freq="Q")

house_prices = house_prices[
    (house_prices["quarter"] >= start_q) & (house_prices["quarter"] <= end_q)
].copy()

rents = rents[
    (rents["quarter"] >= start_q) & (rents["quarter"] <= end_q)
].copy()

leverage = leverage[
    (leverage["quarter"] >= start_q) & (leverage["quarter"] <= end_q)
].copy()

shocks = shocks[
    (shocks["quarter"] >= start_q) & (shocks["quarter"] <= end_q)
].copy()

# --------------------------------------------------
# 5. Merge country-quarter datasets
# --------------------------------------------------
panel = house_prices.merge(
    rents,
    on=["country", "quarter"],
    how="inner"
)

panel = panel.merge(
    leverage,
    on=["country", "quarter"],
    how="inner"
)

# ECB shocks are common across countries, so merge only on quarter
panel = panel.merge(
    shocks,
    on="quarter",
    how="inner"
)

# --------------------------------------------------
# 6. Create price-rent ratio
# --------------------------------------------------
panel["price_rent_ratio"] = panel["house_price_index"] / panel["rent_index"]

# Optional log variables
panel["log_house_price"] = np.log(panel["house_price_index"])
panel["log_rent"] = np.log(panel["rent_index"])
panel["log_price_rent_ratio"] = np.log(panel["price_rent_ratio"])

# --------------------------------------------------
# 7. Sort and inspect
# --------------------------------------------------
panel = panel.sort_values(["country", "quarter"]).reset_index(drop=True)

print("\nFINAL PANEL - first 10 rows")
print(panel.head(10))

print("\nFINAL PANEL SHAPE")
print(panel.shape)

print("\nFINAL PANEL COLUMNS")
print(panel.columns.tolist())

print("\nCOUNTRIES IN FINAL PANEL")
print(sorted(panel["country"].unique()))

print("\nQUARTER RANGE")
print(panel["quarter"].min(), "to", panel["quarter"].max())

print("\nNUMBER OF QUARTERS")
print(panel["quarter"].nunique())

print("\nROWS PER COUNTRY")
print(panel.groupby("country").size())

print("\nMISSING VALUES BY COLUMN")
print(panel.isna().sum())

# --------------------------------------------------
# 8. Save final clean panel to CSV
# --------------------------------------------------
panel_to_save = panel.copy()
panel_to_save["quarter"] = panel_to_save["quarter"].astype(str)

output_file = DATA_DIR / "final_panel_baseline.csv"
panel_to_save.to_csv(output_file, index=False)

print(f"\nSaved clean final panel to: {output_file.resolve()}")