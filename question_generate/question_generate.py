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

from verl.utils.dataset import SPICE_UNIFIED_CHALLENGER_PROMPT

STORAGE_PATH = os.getenv("STORAGE_PATH")

def extract_boxed(text):
    results, i = [], 0
    prefix = r'\boxed{'
    plen = len(prefix)

    while True:
        start = text.find(prefix, i)
        if start == -1:
            break   # no more \boxed{…}

        j = start + plen
        depth = 1
        while j < len(text) and depth:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1

        results.append(text[start + plen : j - 1])
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

def build_prompt(tokenizer, context_text: str, mode: str = "spice") -> str:
    # SPICE paper: Uniformly sample segments of up to 5992 tokens
    MAX_DOC_TOKENS = 5992
    context_tokens = tokenizer.encode(context_text, add_special_tokens=False)

    if len(context_tokens) > MAX_DOC_TOKENS:
        max_start_idx = len(context_tokens) - MAX_DOC_TOKENS
        start_idx = random.randint(0, max_start_idx)
        chunk_tokens = context_tokens[start_idx : start_idx + MAX_DOC_TOKENS]
        context_text = tokenizer.decode(chunk_tokens)

    if mode == "legacy":
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
    else:
        # SPICE challenger JSON模式，与训练保持一致
        chat = [
            {
                "role": "system",
                "content": (
                    "You are an expert exam question setter. "
                    "You must perform task selection and question generation in a single step following the strict guidelines provided. "
                    "Output strictly in JSON format."
                ),
            },
            {
                "role": "user",
                "content": SPICE_UNIFIED_CHALLENGER_PROMPT.format(document=context_text),
            },
        ]

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, add_special_tokens=True
        )
    else:
        prompt = "system: " + chat[0]["content"] + "\n" + "user: " + chat[1]["content"]

    return prompt


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

    if args.corpus_path.endswith(".parquet"):
        corpus_df = pd.read_parquet(args.corpus_path)
    else:
        corpus_df = pd.read_json(
            args.corpus_path,
            lines=args.corpus_path.endswith(".jsonl"),
        )

    replace = len(corpus_df) < args.num_samples
    sampled_df = corpus_df.sample(n=args.num_samples, replace=replace, random_state=int(args.suffix) if args.suffix else 0)
    contexts = sampled_df[args.context_column].fillna("").astype(str).tolist()
    context_ids = sampled_df[args.id_column].fillna("").astype(str).tolist()

    prompts = [build_prompt(tokenizer, ctx, mode=args.mode) for ctx in contexts]

    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    completions: List[RequestOutput] = model.generate(prompts, sampling_params=sample_params)
    results = []
    for idx, completion in enumerate(completions):
        response = completion.outputs[0].text
        question = ""
        answer = ""

        # 优先解析 SPICE JSON
        try:
            obj = json.loads(response)
            if isinstance(obj, dict):
                gen_phase = obj.get("generation_phase", {})
                question = gen_phase.get("question", "") or ""
                answer = gen_phase.get("answer", "") or ""
        except Exception:
            pass

        # 回退到 <question> … </question> + \boxed{}
        if not question or not answer:
            try:
                questions = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
                answers = extract_boxed(response)
                if questions and answers:
                    question = questions[-1].strip()
                    answer = answers[-1].strip()
            except Exception:
                pass

        entry = {
            "question": question if question and answer else response,
            "answer": answer if question and answer else "",
            "score": 0 if question and answer else -1,
            "context": contexts[idx],
            "context_id": context_ids[idx],
            "raw": response,
            "prompt": prompts[idx],
        }
        results.append(entry)
            
    output_dir = os.path.join(STORAGE_PATH, "generated_question")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{args.save_name}_{args.suffix}.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--num_samples", type=int, default=1250, help="Number of samples to generate")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add to the output file")
    parser.add_argument("--save_name", type=str, default="", help="")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus file (parquet/json/jsonl).")
    parser.add_argument("--context_column", type=str, default="text", help="Column containing context text.")
    parser.add_argument("--id_column", type=str, default="id", help="Column containing corpus row id.")
    parser.add_argument("--mode", type=str, default="spice", choices=["spice", "legacy"], help="Prompt/output mode for question generation.")
    args = parser.parse_args()

    main(args)
