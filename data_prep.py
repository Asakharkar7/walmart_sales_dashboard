import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)

def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    features = pd.read_csv(DATA_DIR / "features.csv")
    stores = pd.read_csv(DATA_DIR / "stores.csv")

    for df in (train, features):
        df["Date"] = pd.to_datetime(df["Date"])

    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores, on="Store", how="left")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    out_path = OUT_DIR / "walmart_merged.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved: {out_path} with {len(df):,} rows and {len(df.columns)} columns")

if __name__ == "__main__":
    main()
