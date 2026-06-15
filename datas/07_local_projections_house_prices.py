from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
df = df.sort_values(["country", "quarter"]).reset_index(drop=True)
df["quarter_str"] = df["quarter"].astype(str)

# --------------------------------------------------
# 3. Create lagged leverage and standardized leverage
# --------------------------------------------------
df["leverage_lag"] = df.groupby("country")["leverage_ratio"].shift(1)

mean_lev = df["leverage_lag"].mean()
std_lev = df["leverage_lag"].std()

df["leverage_lag_std"] = (df["leverage_lag"] - mean_lev) / std_lev

# --------------------------------------------------
# 4. Set local projection horizons
# --------------------------------------------------
horizons = list(range(1, 9))   # 1 to 8 quarters ahead

# --------------------------------------------------
# 5. Create cumulative future change in log house prices
# --------------------------------------------------
# For each horizon h:
# y_{i,t,h} = 100 * [log(HPI_{i,t+h}) - log(HPI_{i,t})]
#
# This means:
# cumulative house-price change from quarter t to quarter t+h
# --------------------------------------------------
for h in horizons:
    df[f"lp_dlog_house_price_h{h}"] = (
        df.groupby("country")["log_house_price"].shift(-h) - df["log_house_price"]
    ) * 100

# --------------------------------------------------
# 6. Run one regression per horizon
# --------------------------------------------------
results_list = []
all_summaries = []

for h in horizons:
    dep_var = f"lp_dlog_house_price_h{h}"

    reg_df = df.dropna(subset=[dep_var, "leverage_lag_std", "ecb_shock"]).copy()

    # interaction term
    reg_df["shock_x_lev"] = reg_df["ecb_shock"] * reg_df["leverage_lag_std"]

    # same logic as baseline:
    # - country FE
    # - quarter FE
    # - no separate ecb_shock main effect because it is absorbed by quarter FE
    formula = f"""
    {dep_var} ~ leverage_lag_std + shock_x_lev + C(country) + C(quarter_str)
    """

    model = smf.ols(formula=formula, data=reg_df)

    fit = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_df["country"]}
    )

    beta = fit.params["shock_x_lev"]
    se = fit.bse["shock_x_lev"]
    pval = fit.pvalues["shock_x_lev"]

    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    results_list.append({
        "horizon": h,
        "n_obs": len(reg_df),
        "coef_shock_x_lev": beta,
        "std_error": se,
        "p_value": pval,
        "ci_low_95": ci_low,
        "ci_high_95": ci_high,
        "r_squared": fit.rsquared
    })

    all_summaries.append("\n" + "=" * 80)
    all_summaries.append(f"LOCAL PROJECTION - HOUSE PRICES - HORIZON {h}")
    all_summaries.append("=" * 80)
    all_summaries.append(fit.summary().as_text())

# --------------------------------------------------
# 7. Save results table
# --------------------------------------------------
lp_results = pd.DataFrame(results_list)

lp_results.to_csv("lp_house_prices_results.csv", index=False)

with open("lp_house_prices_summaries.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(all_summaries))

# --------------------------------------------------
# 8. Print compact results
# --------------------------------------------------
print("\nLOCAL PROJECTION RESULTS - HOUSE PRICES")
print(lp_results)

# --------------------------------------------------
# 9. Plot coefficient over horizons
# --------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(lp_results["horizon"], lp_results["coef_shock_x_lev"], marker="o")
plt.fill_between(
    lp_results["horizon"],
    lp_results["ci_low_95"],
    lp_results["ci_high_95"],
    alpha=0.2
)
plt.axhline(0, linewidth=1)

plt.title("Dynamic effect of ECB shock × leverage on house prices")
plt.xlabel("Horizon (quarters ahead)")
plt.ylabel("Coefficient on shock × leverage")
plt.xticks(horizons)
plt.tight_layout()
plt.savefig("lp_house_prices_plot.png", dpi=300)
plt.show()

# --------------------------------------------------
# 10. Print interpretation help
# --------------------------------------------------
print("\nINTERPRETATION GUIDE")
print("Each coefficient shows whether, at a given horizon,")
print("a tighter ECB shock is associated with a more negative cumulative")
print("house-price response in more leveraged countries.")

print("\nIf the coefficient is negative:")
print("- more leveraged countries experience a stronger decline")
print("  (or weaker cumulative growth) in house prices after the shock.")

print("\nSaved files:")
print("- lp_house_prices_results.csv")
print("- lp_house_prices_summaries.txt")
print("- lp_house_prices_plot.png")