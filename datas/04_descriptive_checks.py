from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load final panel
# --------------------------------------------------
DATA_DIR = Path(".")
panel_file = DATA_DIR / "final_panel_baseline.csv"

panel = pd.read_csv(panel_file)

# --------------------------------------------------
# 2. Prepare quarter as datetime for plotting
# --------------------------------------------------
# Convert strings like '2002Q1' into pandas Period, then timestamp
panel["quarter"] = pd.PeriodIndex(panel["quarter"], freq="Q")
panel["quarter_date"] = panel["quarter"].dt.to_timestamp()

# --------------------------------------------------
# 3. Basic summary statistics
# --------------------------------------------------
main_vars = [
    "house_price_index",
    "rent_index",
    "leverage_ratio",
    "ecb_shock",
    "price_rent_ratio"
]

print("\nSUMMARY STATISTICS")
print(panel[main_vars].describe())

print("\nAVERAGE LEVERAGE BY COUNTRY")
print(panel.groupby("country")["leverage_ratio"].mean().sort_values(ascending=False))

print("\nAVERAGE HOUSE PRICE INDEX BY COUNTRY")
print(panel.groupby("country")["house_price_index"].mean().sort_values(ascending=False))

print("\nAVERAGE RENT INDEX BY COUNTRY")
print(panel.groupby("country")["rent_index"].mean().sort_values(ascending=False))

print("\nAVERAGE PRICE-RENT RATIO BY COUNTRY")
print(panel.groupby("country")["price_rent_ratio"].mean().sort_values(ascending=False))

# --------------------------------------------------
# 4. Plot 1: House prices by country
# --------------------------------------------------
plt.figure(figsize=(12, 6))
for country, df_country in panel.groupby("country"):
    plt.plot(df_country["quarter_date"], df_country["house_price_index"], label=country)

plt.title("House Price Index by Country")
plt.xlabel("Quarter")
plt.ylabel("Index")
plt.legend()
plt.tight_layout()
plt.savefig("plot_house_prices.png", dpi=300)
plt.show()

# --------------------------------------------------
# 5. Plot 2: Rents by country
# --------------------------------------------------
plt.figure(figsize=(12, 6))
for country, df_country in panel.groupby("country"):
    plt.plot(df_country["quarter_date"], df_country["rent_index"], label=country)

plt.title("Rent Index by Country")
plt.xlabel("Quarter")
plt.ylabel("Index")
plt.legend()
plt.tight_layout()
plt.savefig("plot_rents.png", dpi=300)
plt.show()

# --------------------------------------------------
# 6. Plot 3: Leverage by country
# --------------------------------------------------
plt.figure(figsize=(12, 6))
for country, df_country in panel.groupby("country"):
    plt.plot(df_country["quarter_date"], df_country["leverage_ratio"], label=country)

plt.title("Household Leverage by Country")
plt.xlabel("Quarter")
plt.ylabel("Debt / Income Ratio")
plt.legend()
plt.tight_layout()
plt.savefig("plot_leverage.png", dpi=300)
plt.show()

# --------------------------------------------------
# 7. Plot 4: ECB shock over time
# --------------------------------------------------
# ECB shock is the same for all countries in each quarter, so keep one observation per quarter
shock_series = panel[["quarter_date", "ecb_shock"]].drop_duplicates().sort_values("quarter_date")

plt.figure(figsize=(12, 6))
plt.plot(shock_series["quarter_date"], shock_series["ecb_shock"])

plt.title("ECB Monetary Policy Shock Over Time")
plt.xlabel("Quarter")
plt.ylabel("ECB Shock")
plt.tight_layout()
plt.savefig("plot_ecb_shock.png", dpi=300)
plt.show()

print("\nPlots saved successfully:")
print("- plot_house_prices.png")
print("- plot_rents.png")
print("- plot_leverage.png")
print("- plot_ecb_shock.png")