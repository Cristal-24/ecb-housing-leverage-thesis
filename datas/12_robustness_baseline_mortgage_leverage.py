from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

# --------------------------------------------------
# 1. Load robustness panel
# --------------------------------------------------
DATA_DIR = Path(".")
panel_file = DATA_DIR / "final_panel_mortgage_leverage.csv"

df = pd.read_csv(panel_file)

# --------------------------------------------------
# 2. Basic preparation
# --------------------------------------------------
df = df.sort_values(["country", "quarter"]).reset_index(drop=True)
df["quarter_str"] = df["quarter"].astype(str)

# --------------------------------------------------
# 3. Create dependent variables
# --------------------------------------------------
# Quarterly log growth rates, multiplied by 100
df["dlog_house_price"] = df.groupby("country")["log_house_price"].diff() * 100
df["dlog_rent"] = df.groupby("country")["log_rent"].diff() * 100
df["dlog_price_rent_ratio"] = df.groupby("country")["log_price_rent_ratio"].diff() * 100

# --------------------------------------------------
# 4. Create lagged mortgage leverage
# --------------------------------------------------
df["mortgage_leverage_lag"] = df.groupby("country")["mortgage_leverage_ratio"].shift(1)

# Standardize lagged mortgage leverage
mean_lev = df["mortgage_leverage_lag"].mean()
std_lev = df["mortgage_leverage_lag"].std()

df["mortgage_leverage_lag_std"] = (
    df["mortgage_leverage_lag"] - mean_lev
) / std_lev

# --------------------------------------------------
# 5. Create interaction term
# --------------------------------------------------
df["shock_x_mortgage_lev"] = df["ecb_shock"] * df["mortgage_leverage_lag_std"]

# --------------------------------------------------
# 6. Helper function to run robustness regression
# --------------------------------------------------
def run_robustness_regression(data, dep_var, output_prefix):
    reg_df = data.dropna(
        subset=[dep_var, "mortgage_leverage_lag_std", "shock_x_mortgage_lev"]
    ).copy()

    print("\n" + "=" * 80)
    print(f"ROBUSTNESS REGRESSION FOR: {dep_var}")
    print("=" * 80)

    print("\nESTIMATION SAMPLE SHAPE")
    print(reg_df.shape)

    print("\nESTIMATION SAMPLE PERIOD")
    print(reg_df["quarter"].min(), "to", reg_df["quarter"].max())

    print("\nCOUNTRIES IN ESTIMATION SAMPLE")
    print(sorted(reg_df["country"].unique()))

    formula = f"""
    {dep_var} ~ mortgage_leverage_lag_std + shock_x_mortgage_lev + C(country) + C(quarter_str)
    """

    model = smf.ols(formula=formula, data=reg_df)

    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_df["country"]}
    )

    print("\nREGRESSION RESULTS")
    print(results.summary())

    # Save coefficient table
    coef_table = pd.DataFrame({
        "variable": results.params.index,
        "coefficient": results.params.values,
        "std_error": results.bse.values,
        "t_stat": results.tvalues.values,
        "p_value": results.pvalues.values
    })

    coef_table.to_csv(f"{output_prefix}_coefficients.csv", index=False)

    # Save full summary
    with open(f"{output_prefix}_summary.txt", "w", encoding="utf-8") as f:
        f.write(results.summary().as_text())

    # Main coefficient
    beta = results.params["shock_x_mortgage_lev"]
    se = results.bse["shock_x_mortgage_lev"]
    pval = results.pvalues["shock_x_mortgage_lev"]

    print("\nMAIN COEFFICIENT OF INTEREST")
    print(f"shock_x_mortgage_lev coefficient = {beta:.6f}")
    print(f"standard error = {se:.6f}")
    print(f"p-value = {pval:.6f}")

    if beta < 0:
        print("\nInterpretation: tighter ECB shocks are associated with a lower value")
        print("of the dependent variable in countries with higher mortgage leverage.")
    elif beta > 0:
        print("\nInterpretation: tighter ECB shocks are associated with a higher value")
        print("of the dependent variable in countries with higher mortgage leverage.")
    else:
        print("\nInterpretation: the interaction effect is essentially zero.")

    print("\nSaved files:")
    print(f"- {output_prefix}_coefficients.csv")
    print(f"- {output_prefix}_summary.txt")

    return {
        "dependent_variable": dep_var,
        "n_obs": len(reg_df),
        "period_start": reg_df["quarter"].min(),
        "period_end": reg_df["quarter"].max(),
        "coef_shock_x_mortgage_lev": beta,
        "std_error": se,
        "p_value": pval,
        "r_squared": results.rsquared
    }

# --------------------------------------------------
# 7. Run robustness regressions
# --------------------------------------------------
compact_results = []

compact_results.append(
    run_robustness_regression(
        data=df,
        dep_var="dlog_house_price",
        output_prefix="robustness_mortgage_house_price"
    )
)

compact_results.append(
    run_robustness_regression(
        data=df,
        dep_var="dlog_rent",
        output_prefix="robustness_mortgage_rent"
    )
)

compact_results.append(
    run_robustness_regression(
        data=df,
        dep_var="dlog_price_rent_ratio",
        output_prefix="robustness_mortgage_prratio"
    )
)

# --------------------------------------------------
# 8. Save compact results table
# --------------------------------------------------
compact_results_df = pd.DataFrame(compact_results)
compact_results_df.to_csv("robustness_mortgage_baseline_results.csv", index=False)

print("\n" + "=" * 80)
print("COMPACT ROBUSTNESS RESULTS")
print("=" * 80)
print(compact_results_df)

print("\nSaved file:")
print("- robustness_mortgage_baseline_results.csv")