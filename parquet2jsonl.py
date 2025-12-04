import pandas as pd
import argparse
import json

def parquet_to_jsonl(parquet_file, jsonl_file):
    print(f"📦 Loading parquet: {parquet_file}")
    df = pd.read_parquet(parquet_file, engine="pyarrow")

    print(f"🔍 Loaded {len(df)} rows, writing to JSONL...")
    print(f"📤 Saving to jsonl: {jsonl_file}")

    with open(jsonl_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            json_line = json.dumps(row.to_dict(), ensure_ascii=False)
            f.write(json_line + "\n")

    print("🎉 Done!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input .parquet file")
    parser.add_argument("--output", type=str, required=True, help="Output .jsonl file")
    args = parser.parse_args()

    parquet_to_jsonl(args.input, args.output)

if __name__ == "__main__":
    main()
