import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Inspect a Parquet file.")
    parser.add_argument("--file", type=str, required=True, help="Path to parquet file")
    parser.add_argument("--head", type=int, default=5, help="Show first N rows")
    args = parser.parse_args()

    print(f"\n📦 Loading parquet file: {args.file}")
    df = pd.read_parquet(args.file)

    print("\n🧱 Columns:")
    print(df.columns.tolist())

    print(f"\n🔍 First {args.head} rows:")
    print(df.head(args.head))

    print("\n✨ Shape (rows, columns):", df.shape)

    print("\nDone.\n")

if __name__ == "__main__":
    main()
