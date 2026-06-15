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
df = df.sort_values(["country", "quarter"]).reset_index(drop=True)
df["quarter_str"] = df["quarter"].astype(str)

# --------------------------------------------------
# 3. Create dependent variables
# --------------------------------------------------
# Quarterly rent growth
df["dlog_rent"] = df.groupby("country")["log_rent"].diff() * 100

# Quarterly growth in the log price-rent ratio
df["dlog_price_rent_ratio"] = df.groupby("country")["log_price_rent_ratio"].diff() * 100

# --------------------------------------------------
# 4. Create lagged leverage
# --------------------------------------------------
df["leverage_lag"] = df.groupby("country")["leverage_ratio"].shift(1)

mean_lev = df["leverage_lag"].mean()
std_lev = df["leverage_lag"].std()

df["leverage_lag_std"] = (df["leverage_lag"] - mean_lev) / std_lev

# --------------------------------------------------
# 5. Interaction term
# --------------------------------------------------
df["shock_x_lev"] = df["ecb_shock"] * df["leverage_lag_std"]

# --------------------------------------------------
# 6. Helper function to run regression
# --------------------------------------------------
def run_baseline_regression(data, dep_var, output_prefix):
    reg_df = data.dropna(subset=[dep_var, "leverage_lag_std", "shock_x_lev"]).copy()

    print("\n" + "=" * 80)
    print(f"BASELINE REGRESSION FOR: {dep_var}")
    print("=" * 80)

    print("\nESTIMATION SAMPLE SHAPE")
    print(reg_df.shape)

    print("\nESTIMATION SAMPLE PERIOD")
    print(reg_df["quarter"].min(), "to", reg_df["quarter"].max())

    print("\nCOUNTRIES IN ESTIMATION SAMPLE")
    print(sorted(reg_df["country"].unique()))

    formula = f"""
    {dep_var} ~ leverage_lag_std + shock_x_lev + C(country) + C(quarter_str)
    """

    model = smf.ols(formula=formula, data=reg_df)

    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_df["country"]}
    )

    print("\nREGRESSION RESULTS")
    print(results.summary())

    # Save coefficients
    coef_table = pd.DataFrame({
        "variable": results.params.index,
        "coefficient": results.params.values,
        "std_error": results.bse.values,
        "t_stat": results.tvalues.values,
        "p_value": results.pvalues.values
    })

    coef_table.to_csv(f"{output_prefix}_coefficients.csv", index=False)

    # Save summary
    with open(f"{output_prefix}_summary.txt", "w", encoding="utf-8") as f:
        f.write(results.summary().as_text())

    # Print main coefficient clearly
    beta = results.params["shock_x_lev"]
    pval = results.pvalues["shock_x_lev"]

    print("\nMAIN COEFFICIENT OF INTEREST")
    print(f"shock_x_lev coefficient = {beta:.6f}")
    print(f"p-value = {pval:.6f}")

    if beta < 0:
        print("\nInterpretation: tighter ECB shocks are associated with a lower value of the dependent variable")
        print("in more leveraged countries.")
    elif beta > 0:
        print("\nInterpretation: tighter ECB shocks are associated with a higher value of the dependent variable")
        print("in more leveraged countries.")
    else:
        print("\nInterpretation: the interaction effect is essentially zero in this baseline.")

    print("\nSaved files:")
    print(f"- {output_prefix}_coefficients.csv")
    print(f"- {output_prefix}_summary.txt")

    return results

# --------------------------------------------------
# 7. Run regression for rents
# --------------------------------------------------
rent_results = run_baseline_regression(
    data=df,
    dep_var="dlog_rent",
    output_prefix="baseline_rent_regression"
)

# --------------------------------------------------
# 8. Run regression for price-rent ratio
# --------------------------------------------------
prratio_results = run_baseline_regression(
    data=df,
    dep_var="dlog_price_rent_ratio",
    output_prefix="baseline_prratio_regression"
)