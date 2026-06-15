from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------
DATA_DIR = Path(".")

house_prices_file = DATA_DIR / "raw_data  house_prices.csv"
rents_file = DATA_DIR / "rents_raw_ecb_hicp_monthly.csv"
leverage_file = DATA_DIR / "Loans granted to households, ratio of adjusted gross disposable income.csv"
shocks_file = DATA_DIR / "shocks_ecb_mpd_me_m (1).csv"

# --------------------------------------------------
# 2. Common country list
# --------------------------------------------------
countries = ["AT", "BE", "FI", "FR", "DE", "GR", "IE", "IT", "NL", "PT", "ES"]

# --------------------------------------------------
# 3. Load raw files
# --------------------------------------------------
house_prices_raw = pd.read_csv(house_prices_file, skiprows=7)
rents_raw = pd.read_csv(rents_file)
leverage_raw = pd.read_csv(leverage_file)
shocks_raw = pd.read_csv(shocks_file)

# --------------------------------------------------
# 4. Clean HOUSE PRICES (BIS)
# --------------------------------------------------
house_prices = house_prices_raw.copy()

house_prices = house_prices[
    house_prices["REF_AREA:Reference area"].isin([f"{c}:{name}" for c, name in [
        ("AT", "Austria"), ("BE", "Belgium"), ("FI", "Finland"), ("FR", "France"),
        ("DE", "Germany"), ("GR", "Greece"), ("IE", "Ireland"), ("IT", "Italy"),
        ("NL", "Netherlands"), ("PT", "Portugal"), ("ES", "Spain")
    ]])
].copy()

# extract country code
house_prices["country"] = house_prices["REF_AREA:Reference area"].str.split(":").str[0]

# date to quarter
house_prices["date"] = pd.to_datetime(house_prices["TIME_PERIOD:Period"])
house_prices["quarter"] = house_prices["date"].dt.to_period("Q")

# keep only what matters
house_prices = house_prices[["country", "quarter", "OBS_VALUE:Value"]].copy()
house_prices = house_prices.rename(columns={"OBS_VALUE:Value": "house_price_index"})

# --------------------------------------------------
# 5. Clean RENTS (ECB HICP monthly)
# --------------------------------------------------
rents = rents_raw.copy()

# date column
rents["date"] = pd.to_datetime(rents["DATE"])
rents["quarter"] = rents["date"].dt.to_period("Q")

# keep only country columns we need
rent_cols = {}
for col in rents.columns:
    for c in countries:
        if f"HICP.M.{c}." in col:
            rent_cols[col] = c

rents = rents[["quarter"] + list(rent_cols.keys())].copy()
rents = rents.rename(columns=rent_cols)

# reshape to long
rents = rents.groupby("quarter")[countries].mean().reset_index()
rents = rents.melt(id_vars="quarter", var_name="country", value_name="rent_index")

# --------------------------------------------------
# 6. Clean LEVERAGE (already quarterly)
# --------------------------------------------------
leverage = leverage_raw.copy()

leverage["date"] = pd.to_datetime(leverage["DATE"])
leverage["quarter"] = leverage["date"].dt.to_period("Q")

lev_cols = {}
for col in leverage.columns:
    for c in countries:
        if f".{c}." in col:
            lev_cols[col] = c

leverage = leverage[["quarter"] + list(lev_cols.keys())].copy()
leverage = leverage.rename(columns=lev_cols)

# reshape to long
leverage = leverage.melt(id_vars="quarter", var_name="country", value_name="leverage_ratio")

# --------------------------------------------------
# 7. Clean ECB SHOCKS (monthly -> quarterly sum)
# --------------------------------------------------
shocks = shocks_raw.copy()

shocks["date"] = pd.to_datetime(
    shocks["year"].astype(str) + "-" + shocks["month"].astype(str).str.zfill(2) + "-01"
)
shocks["quarter"] = shocks["date"].dt.to_period("Q")

shocks = shocks.groupby("quarter", as_index=False)["MP_median"].sum()
shocks = shocks.rename(columns={"MP_median": "ecb_shock"})

# --------------------------------------------------
# 8. Restrict sample from 2002Q1 onward
# --------------------------------------------------
start_q = pd.Period("2002Q1", freq="Q")

house_prices = house_prices[house_prices["quarter"] >= start_q].copy()
rents = rents[rents["quarter"] >= start_q].copy()
leverage = leverage[leverage["quarter"] >= start_q].copy()
shocks = shocks[shocks["quarter"] >= start_q].copy()

# --------------------------------------------------
# 9. Print checks
# --------------------------------------------------
print("\nHOUSE PRICES cleaned:")
print(house_prices.head())
print(house_prices.shape)

print("\nRENTS cleaned:")
print(rents.head())
print(rents.shape)

print("\nLEVERAGE cleaned:")
print(leverage.head())
print(leverage.shape)

print("\nECB SHOCKS cleaned:")
print(shocks.head())
print(shocks.shape)

# --------------------------------------------------
# 10. Save cleaned intermediate files
# --------------------------------------------------
house_prices.to_csv("clean_house_prices.csv", index=False)
rents.to_csv("clean_rents.csv", index=False)
leverage.to_csv("clean_leverage.csv", index=False)
shocks.to_csv("clean_ecb_shocks.csv", index=False)

print("\nCleaned files saved successfully.")