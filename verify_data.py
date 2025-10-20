import pandas as pd
from pathlib import Path

DATA_PATH = Path("artifacts") / "walmart_merged.parquet"

def main():
    df = pd.read_parquet(DATA_PATH)

    print("\n✅ File Loaded Successfully!")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    print("📋 Columns:")
    print(df.columns.tolist())
    print("\n🔍 Sample Data:")
    print(df.head(10))

    print("\n🧮 Missing Values per Column:")
    print(df.isna().sum())

if __name__ == "__main__":
    main()
