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

from scripts.prompts import get_free_form_question_challenger_prompt

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

def truncate_context(tokenizer, context_text: str, max_doc_tokens: int) -> str:
    """Uniformly sample a window if the context exceeds max_doc_tokens."""
    context_tokens = tokenizer.encode(context_text, add_special_tokens=False)
    if len(context_tokens) > max_doc_tokens:
        max_start_idx = len(context_tokens) - max_doc_tokens
        start_idx = random.randint(0, max_start_idx)
        chunk_tokens = context_tokens[start_idx : start_idx + max_doc_tokens]
        context_text = tokenizer.decode(chunk_tokens)
    return context_text


def build_prompt(
    tokenizer,
    context_text: str,
    mode: str = "free_form",
    answer_type: str = "integer",
    max_doc_tokens: int = 4096,
) -> str:
    """
    Build the questioner prompt using the shared free-form challenger template
    (same as scripts/prompts.py + verl/utils/dataset.py). Mode is kept for
    backward compatibility but only "free_form"/"spice" are allowed.
    """
    context_text = truncate_context(tokenizer, context_text, max_doc_tokens)

    if mode not in {"spice", "free_form"}:
        raise ValueError(f"Unsupported mode: {mode}")

    prompt_text = get_free_form_question_challenger_prompt(
        text=context_text,
        answer_type=answer_type,
    )
    messages = [{"role": "user", "content": prompt_text}]

    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt_text


def _normalize_answer_text(answer):
    if isinstance(answer, list):
        answer = answer[-1] if answer else ""
    if isinstance(answer, (int, float, bool)):
        return str(answer)
    if not isinstance(answer, str):
        return ""
    return answer.strip()


def parse_question_answer(response: str):
    """Extract question/answer from structured JSON or legacy formats."""
    question = ""
    answer = ""

    # Structured JSON (SPICE/free-form challenger or MCQ)
    try:
        obj = json.loads(response)
        if isinstance(obj, dict):
            gen_phase = obj.get("generation_phase")
            if isinstance(gen_phase, dict):
                question = _normalize_answer_text(gen_phase.get("question", "")) or question
                answer = _normalize_answer_text(gen_phase.get("answer", "")) or answer

            question = _normalize_answer_text(obj.get("question", "")) or question
            answer = _normalize_answer_text(obj.get("answer", "")) or answer

            if not question:
                for key in ("exam_question", "multiple_choice_question"):
                    question = _normalize_answer_text(obj.get(key, ""))
                    if question:
                        break

            if not answer:
                for key in ("correct_answer", "final_answer", "identified_answer", "ground_truth"):
                    answer = _normalize_answer_text(obj.get(key, ""))
                    if answer:
                        break
    except Exception:
        pass

    # Regex fallback for JSON-ish strings
    if not question:
        match = re.search(r'"question"\s*:\s*"([^"]+)"', response, re.DOTALL)
        if match:
            question = _normalize_answer_text(match.group(1))

    if not answer:
        match = re.search(r'"answer"\s*:\s*"([^"]+)"', response, re.DOTALL)
        if match:
            answer = _normalize_answer_text(match.group(1))
        else:
            match = re.search(r'"answer"\s*:\s*([-+]?\d+(?:\.\d+)?)', response)
            if match:
                answer = _normalize_answer_text(match.group(1))

    # Legacy <question>...</question> + \boxed{} format
    if not question or not answer:
        try:
            questions = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
            answers = extract_boxed(response)
            if questions and not question:
                question = _normalize_answer_text(questions[-1])
            if answers and not answer:
                answer = _normalize_answer_text(answers[-1])
        except Exception:
            pass

    return question, answer


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

    prompts = [
        build_prompt(
            tokenizer,
            ctx,
            mode=args.mode,
            answer_type=args.answer_type,
            max_doc_tokens=args.max_doc_tokens,
        )
        for ctx in contexts
    ]

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
        question, answer = parse_question_answer(response)

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
    parser.add_argument(
        "--mode",
        type=str,
        default="free_form",
        choices=["free_form", "spice"],
        help="Prompt/output mode for question generation. free_form/spice both use the shared challenger prompt.",
    )
    parser.add_argument(
        "--answer_type",
        type=str,
        default="integer",
        help="Answer type placeholder used in the free-form challenger prompt.",
    )
    parser.add_argument(
        "--max_doc_tokens",
        type=int,
        default=4096,
        help="Maximum number of context tokens to feed into the prompt.",
    )
    args = parser.parse_args()

    main(args)
