from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. Folder path
# --------------------------------------------------
DATA_DIR = Path(".")

# --------------------------------------------------
# 2. Helper function to find files by pattern
# --------------------------------------------------
def find_file(pattern: str) -> Path:
    matches = list(DATA_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    if len(matches) > 1:
        print(f"[WARNING] Multiple files found for pattern {pattern}:")
        for m in matches:
            print("   ", m.name)
        print(f"Using the first one: {matches[0].name}")
    return matches[0]

# --------------------------------------------------
# 3. Find your files
# --------------------------------------------------
house_prices_file = find_file("*house*price*.csv")
rents_file = find_file("*rent*.csv")
leverage_file = find_file("*Loans granted to households*.csv")
shocks_file = find_file("*shocks_ecb_mpd_me_m*.csv")

print("\nFiles found:")
print("House prices :", house_prices_file.name)
print("Rents        :", rents_file.name)
print("Leverage     :", leverage_file.name)
print("ECB shocks   :", shocks_file.name)

# --------------------------------------------------
# 4. Load the files
# --------------------------------------------------
# BIS file has metadata rows before the actual table
house_prices = pd.read_csv(house_prices_file, skiprows=7)

# These 3 can be read normally
rents = pd.read_csv(rents_file)
leverage = pd.read_csv(leverage_file)
shocks = pd.read_csv(shocks_file)

# --------------------------------------------------
# 5. Helper function to inspect a dataframe
# --------------------------------------------------
def inspect_df(name: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print(f"{name}")
    print("=" * 70)
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())

# --------------------------------------------------
# 6. Inspect all datasets
# --------------------------------------------------
inspect_df("HOUSE PRICES", house_prices)
inspect_df("RENTS", rents)
inspect_df("LEVERAGE", leverage)
inspect_df("ECB SHOCKS", shocks)