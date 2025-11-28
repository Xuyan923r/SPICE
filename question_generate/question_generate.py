import argparse
import json
import os
import random
from typing import List

import pandas as pd
import regex as re
import torch
import vllm
from transformers import AutoTokenizer
from vllm.outputs import RequestOutput

STORAGE_PATH = os.getenv("STORAGE_PATH")


# ============================================================
# New: Unified loader that supports directories + parquet/json/jsonl
# ============================================================
def load_corpus(corpus_path, context_column, id_column):
    # ------------ Case 1: corpus_path is a directory ----------------
    if os.path.isdir(corpus_path):
        print(f"[INFO] corpus_path is a directory, scanning: {corpus_path}")

        files = []
        for f in os.listdir(corpus_path):
            if f.endswith(".parquet") or f.endswith(".json") or f.endswith(".jsonl"):
                files.append(os.path.join(corpus_path, f))

        if not files:
            raise ValueError(f"No parquet/json/jsonl files found under: {corpus_path}")

        dfs = []
        for f in files:
            print(f"[INFO] Loading: {f}")

            if f.endswith(".parquet"):
                dfs.append(pd.read_parquet(f))

            elif f.endswith(".json"):
                dfs.append(pd.read_json(f))

            elif f.endswith(".jsonl"):
                dfs.append(pd.read_json(f, lines=True))

        corpus_df = pd.concat(dfs, ignore_index=True)

    # ------------ Case 2: corpus_path is a single file -------------
    else:
        print(f"[INFO] Loading single corpus file: {corpus_path}")

        if corpus_path.endswith(".parquet"):
            corpus_df = pd.read_parquet(corpus_path)

        elif corpus_path.endswith(".json"):
            corpus_df = pd.read_json(corpus_path)

        elif corpus_path.endswith(".jsonl"):
            corpus_df = pd.read_json(corpus_path, lines=True)

        else:
            raise ValueError(f"Unsupported corpus file type: {corpus_path}")

    # ------------ Check required columns ----------------
    if context_column not in corpus_df.columns:
        raise ValueError(
            f"Column `{context_column}` not found in corpus. "
            f"Available columns: {list(corpus_df.columns)}"
        )

    if id_column not in corpus_df.columns:
        print(f"[WARN] Column `{id_column}` not found, auto-generating IDs.")
        corpus_df[id_column] = range(len(corpus_df))

    return corpus_df


# ============================================================
# Existing utils
# ============================================================
def extract_boxed(text):
    results, i = [], 0
    prefix = r'\boxed{'
    plen = len(prefix)

    while True:
        start = text.find(prefix, i)
        if start == -1:
            break

        j = start + plen
        depth = 1
        while j < len(text) and depth:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1

        results.append(text[start + plen: j - 1])
        i = j

    return results


def get_response_mask(response_ids, eos_token_id, dtype):
    batch_size, seq_len = response_ids.shape
    mask = torch.ones((batch_size, seq_len), dtype=dtype)
    for i in range(batch_size):
        for j in range(seq_len):
            if response_ids[i][j] == eos_token_id:
                mask[i][j:] = 0
                break
    return mask


def build_prompt(tokenizer, context_text: str) -> str:
    MAX_DOC_TOKENS = 5992
    context_tokens = tokenizer.encode(context_text, add_special_tokens=False)

    # Random chunking
    if len(context_tokens) > MAX_DOC_TOKENS:
        max_start_idx = len(context_tokens) - MAX_DOC_TOKENS
        start_idx = random.randint(0, max_start_idx)
        chunk_tokens = context_tokens[start_idx: start_idx + MAX_DOC_TOKENS]
        context_text = tokenizer.decode(chunk_tokens)

    chat = [
        {
            "role": "system",
            "content": (
                "You are an expert competition-math problem setter.\n"
                "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. "
                "The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. "
                "Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. "
                "Avoid re-using textbook clichés or famous contest problems.\n"
                "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                "<question>\n"
                "{The full problem statement on one or more lines}\n"
                "</question>\n\n"
                r"\boxed{final_answer}"
                "\n\n"
                "Do NOT output anything else—no explanations, no extra markup."
            ),
        },
        {
            "role": "user",
            "content": (
                "Here is some reference material that you must base the new question on. "
                "Read it carefully and ensure the generated problem is grounded in this material:\n\n"
                f"{context_text}\n\n"
                "Generate one new, challenging reasoning question now. "
                "Remember to format the output exactly as instructed."
            ),
        },
    ]

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, add_special_tokens=True
        )
    else:
        prompt = "system: " + chat[0]["content"] + "\n" + "user: " + chat[1]["content"]

    return prompt


# ============================================================
# main()
# ============================================================
def main(args):
    if args.suffix:
        random.seed(int(args.suffix))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        seed=int(args.suffix) if args.suffix else 0,
    )

    if args.corpus_path is None:
        raise ValueError("Please specify --corpus_path to provide context for question generation.")

    # --------- NEW: unified loader -----------
    corpus_df = load_corpus(args.corpus_path, args.context_column, args.id_column)

    # Sampling
    replace = len(corpus_df) < args.num_samples
    sampled_df = corpus_df.sample(
        n=args.num_samples,
        replace=replace,
        random_state=int(args.suffix) if args.suffix else 0,
    )
    contexts = sampled_df[args.context_column].fillna("").astype(str).tolist()
    context_ids = sampled_df[args.id_column].astype(str).tolist()

    prompts = [build_prompt(tokenizer, ctx) for ctx in contexts]

    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    completions: List[RequestOutput] = model.generate(prompts, sampling_params=sample_params)

    # Save results
    results = []
    for idx, completion in enumerate(completions):
        response = completion.outputs[0].text
        try:
            questions = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
            answers = extract_boxed(response)

            if questions and answers:
                question = questions[-1].strip()
                answer = answers[-1].strip()
                score = 0
            else:
                question = response
                answer = ""
                score = -1

        except Exception:
            question = response
            answer = ""
            score = -1

        results.append(
            {
                "question": question,
                "answer": answer,
                "score": score,
                "context": contexts[idx],
                "context_id": context_ids[idx],
            }
        )

    output_dir = os.path.join(STORAGE_PATH, "generated_question")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{args.save_name}_{args.suffix}.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[INFO] Saved generated questions to: {out_path}")


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--num_samples", type=int, default=1250)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--context_column", type=str, default="text")
    parser.add_argument("--id_column", type=str, default="id")

    args = parser.parse_args()
    main(args)
