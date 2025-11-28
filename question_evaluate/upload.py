import argparse
import json
import os
from typing import List

import matplotlib.pyplot as plt
from datasets import Dataset

STORAGE_PATH = os.getenv("STORAGE_PATH")


def load_results(experiment_name: str) -> List[dict]:
    results = []
    for i in range(8):
        path = f"{STORAGE_PATH}/generated_question/{experiment_name}_{i}_results.json"
        try:
            with open(path, "r") as f:
                data = json.load(f)
                results.extend(data)
        except Exception:
            print(f"File {experiment_name}_{i}_results.json not found")
            continue
    return results


def cleanup_results(experiment_name: str):
    for i in range(8):
        path = f"{STORAGE_PATH}/generated_question/{experiment_name}_{i}_results.json"
        try:
            os.remove(path)
        except Exception:
            print(f"File {experiment_name}_{i}_results.json not found")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_score", type=float, default=0.7)
    parser.add_argument("--min_score", type=float, default=0.3)
    parser.add_argument("--experiment_name", type=str, default="Qwen_Qwen3-4B-Base_all")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save the locally filtered dataset (default: STORAGE_PATH/solver_data/<experiment_name>).",
    )
    parser.add_argument(
        "--keep_tmp",
        action="store_true",
        help="Keep intermediate *_results.json files instead of deleting them.",
    )
    args = parser.parse_args()

    print(STORAGE_PATH)

    datas = load_results(args.experiment_name)
    scores = [data["score"] for data in datas]

    # Print and visualize score distribution for quick inspection.
    plt.hist(scores, bins=11)
    plt.savefig("scores_distribution.png")

    # Filter usable samples into local parquet dataset.
    filtered_datas = [
        {"problem": data["question"], "answer": data["answer"], "score": data["score"]}
        for data in datas
        if data["score"] >= args.min_score
        and data["score"] <= args.max_score
        and data["answer"] not in ("", "None")
    ]
    print(f"Filtered samples: {len(filtered_datas)}")

    if filtered_datas:
        train_dataset = Dataset.from_list(filtered_datas)
        output_dir = args.output_dir or os.path.join(STORAGE_PATH, "solver_data", args.experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "train.parquet")
        train_dataset.to_parquet(output_file)
        print(f"Saved local dataset to: {output_file}")
    else:
        print("No samples passed the score/answer filters; no dataset written.")

    if not args.keep_tmp:
        cleanup_results(args.experiment_name)
