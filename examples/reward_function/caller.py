# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import regex as re
from typing import Dict, List
import json
from mathruler.grader import extract_boxed_content, grade_answer
import os
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

STORAGE_PATH = os.getenv("STORAGE_PATH")
QUESTIONER_DUMP_DIR = os.getenv("QUESTIONER_DUMP_DIR")
QUESTIONER_DUMP_FILE = os.getenv("QUESTIONER_DUMP_FILE")
QUESTIONER_DEBUG_LOG = os.getenv("QUESTIONER_DEBUG_LOG")

def _parse_http_timeout():
    """
    Convert env CALLER_HTTP_TIMEOUT to a timeout value.
    - Empty/invalid/<=0/"none"/"off" -> None (no timeout)
    - Otherwise float seconds.
    """
    raw = os.getenv("CALLER_HTTP_TIMEOUT", "").strip().lower()
    if raw in ("", "none", "no", "off", "disable"):
        return None
    try:
        val = float(raw)
        return val if val > 0 else None
    except Exception:
        return None

HTTP_TIMEOUT = _parse_http_timeout()

def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"
def split_list(lst, n=4):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

def fetch(index, path, timeout):
    if timeout is None:
        response = requests.get(f"http://0.0.0.0:{5000+index}/hello?name={path}")
    else:
        response = requests.get(f"http://0.0.0.0:{5000+index}/hello?name={path}", timeout=timeout)
    print(response)
    return True

def generate_results(data):
    datas = split_list(data,4)
    random_names = [generate_temp_filename(prefix=f"temp_{i}", suffix=".json") for i in range(4)]
    for i in range(4):
        with open(random_names[i],'w') as f:
            json.dump(datas[i],f,indent=4)

    final_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, i, random_names[i], HTTP_TIMEOUT) for i in range(4)]

        for future in as_completed(futures):
            print(future.result())

    for i in range(4):
        with open(random_names[i].replace('.json','_results.json'),'r') as f:
            final_results.extend(json.load(f))

    return final_results

def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def _normalize_answer_text(answer):
    """Normalize answers to plain strings for downstream grading."""
    if isinstance(answer, list):
        answer = answer[-1] if answer else ""
    if isinstance(answer, (int, float, bool)):
        return str(answer)
    if not isinstance(answer, str):
        return ""
    return answer.strip()


def _extract_question_answer(predict: str):
    """Parse question & answer from structured JSON or legacy text outputs."""
    question = ""
    answer = ""

    # Structured JSON (SPICE challenger or free-form prompt)
    try:
        obj = json.loads(predict)
        if isinstance(obj, dict):
            gen_phase = obj.get("generation_phase")
            if isinstance(gen_phase, dict):
                question = _normalize_answer_text(gen_phase.get("question", "")) or question
                answer = _normalize_answer_text(gen_phase.get("answer", "")) or answer

            if not question:
                question = _normalize_answer_text(obj.get("question", "")) or question
            if not answer:
                answer = _normalize_answer_text(obj.get("answer", "")) or answer

            if not answer:
                for key in ("correct_answer", "final_answer", "identified_answer"):
                    answer = _normalize_answer_text(obj.get(key, ""))
                    if answer:
                        break
    except Exception:
        pass

    if not question:
        match = re.search(r'"question"\s*:\s*"([^"]+)"', predict, re.DOTALL)
        if match:
            question = _normalize_answer_text(match.group(1))

    if not answer:
        match = re.search(r'"answer"\s*:\s*"([^"]+)"', predict, re.DOTALL)
        if match:
            answer = _normalize_answer_text(match.group(1))
        else:
            match = re.search(r'"answer"\s*:\s*([-+]?\d+(?:\.\d+)?)', predict)
            if match:
                answer = _normalize_answer_text(match.group(1))

    # Legacy <question>...</question> + \boxed{} format
    if not question or not answer:
        try:
            questions = re.findall(r"<question>(.*?)</question>", predict, re.DOTALL)
            answers = extract_boxed_content(predict)
            if questions and not question:
                question = _normalize_answer_text(questions[-1])
            if answers and not answer:
                answer = _normalize_answer_text(answers[-1])
        except Exception:
            pass

    return question, answer


def _resolve_debug_path():
    """
    Returns debug_path for raw & parsed questioner outputs.
    Follows QUESTIONER_DUMP_DIR/QUESTIONER_DUMP_FILE if set, else QUESTIONER_DEBUG_LOG override.
    """
    debug_path = None

    if QUESTIONER_DUMP_DIR or QUESTIONER_DUMP_FILE:
        if QUESTIONER_DUMP_FILE:
            base_dir = os.path.dirname(QUESTIONER_DUMP_FILE)
            if base_dir:
                os.makedirs(base_dir, exist_ok=True)
            debug_path = os.path.join(base_dir, "questioner_debug.jsonl")
        else:
            os.makedirs(QUESTIONER_DUMP_DIR, exist_ok=True)
            debug_path = os.path.join(QUESTIONER_DUMP_DIR, "questioner_debug.jsonl")

    if QUESTIONER_DEBUG_LOG:
        debug_path = QUESTIONER_DEBUG_LOG
        if debug_path:
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)

    return debug_path


def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1, file_path: str = "") -> List[Dict[str, float]]:
    results = []
    debug_entries = []
    with open('test.json','w') as f:
        json.dump(predicts,f,indent=4)
    for predict in predicts:
        question, answer = _extract_question_answer(predict)
        results.append({"question": question, "answer": answer})
        debug_entries.append({
            "raw_predict": predict,
            "parsed_question": question,
            "parsed_answer": answer,
            "run_id": os.getenv("RUN_ID"),
            "timestamp": time.time(),
        })

    final_results = generate_results(results)
    scores = [{"overall": min(item["score"],1-item["score"]) if item['question'] else -1,"format": 1 if item['question'] else 0,"accuracy": 1 if item['answer'] else 0} for item in final_results]

    debug_path = _resolve_debug_path()
    if debug_path and debug_entries:
        try:
            with open(debug_path, "a", encoding="utf-8") as f:
                for entry in debug_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[dump] Failed to dump questioner debug outputs: {e}")

    return scores
