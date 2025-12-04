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
import math
from mathruler.grader import extract_boxed_content, grade_answer
import os
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering
import numpy as np
STORAGE_PATH = os.getenv("STORAGE_PATH","/apdcephfs_sh2/share_300000800/user/chengchuang")
QUESTIONER_DUMP_DIR = os.getenv("QUESTIONER_DUMP_DIR")
QUESTIONER_DUMP_FILE = os.getenv("QUESTIONER_DUMP_FILE")
QUESTIONER_DEBUG_LOG = os.getenv("QUESTIONER_DEBUG_LOG")
def _bleu_distance_matrix(sentences):
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    return dist

def cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    if not problems:
        return []
    print('start clustering')
    start_time = time.time()
    dist_mat = _bleu_distance_matrix(problems)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'end clustering, time: {time.time() - start_time}')
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions

def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"
def split_list(lst, n=4):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

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
QUESTIONER_DUMP_DIR = os.getenv("QUESTIONER_DUMP_DIR")
QUESTIONER_DUMP_FILE = os.getenv("QUESTIONER_DUMP_FILE")
QUESTIONER_DEBUG_LOG = os.getenv("QUESTIONER_DEBUG_LOG")

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
        # os.remove(random_names[i].replace('.json','_results.json'))
    for i in range(4):
        os.remove(random_names[i].replace('.json','_results.json'))
    return final_results

def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


GAUSSIAN_TARGET = 0.25
GAUSSIAN_VARIANCE = 0.01  # sigma^2 in the paper's formulation
INVALID_QUESTION_PENALTY = -1.0


def _normalize_answer_text(answer):
    """Best-effort conversion of boxed answers to comparable strings."""
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

    # 1) Structured JSON (SPICE challenger or free-form prompt)
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

    # 2) Legacy <question>...</question> + \boxed{} format
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


def _resolve_dump_paths():
    """
    Returns (dump_path, debug_path) based on env configuration.
    - dump_path: existing log path for rewards (unchanged behavior)
    - debug_path: new JSONL log for raw & parsed questioner outputs
    """
    dump_path = None
    debug_path = None

    if QUESTIONER_DUMP_DIR or QUESTIONER_DUMP_FILE:
        if QUESTIONER_DUMP_FILE:
            dump_path = QUESTIONER_DUMP_FILE
            base_dir = os.path.dirname(dump_path)
            os.makedirs(base_dir, exist_ok=True)
            debug_path = os.path.join(base_dir, "questioner_debug.jsonl")
        else:
            os.makedirs(QUESTIONER_DUMP_DIR, exist_ok=True)
            dump_path = os.path.join(QUESTIONER_DUMP_DIR, "all_results.jsonl")
            debug_path = os.path.join(QUESTIONER_DUMP_DIR, "questioner_debug.jsonl")

    if QUESTIONER_DEBUG_LOG:
        # Allow explicit override
        debug_path = QUESTIONER_DEBUG_LOG
        if debug_path:
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)

    return dump_path, debug_path


def _reasoner_correctness(reasoner_outputs, golden_answer: str) -> List[float]:
    normalized_gold = _normalize_answer_text(golden_answer)
    if not normalized_gold:
        return []

    indicators: List[float] = []
    for output in reasoner_outputs or []:
        candidate = _normalize_answer_text(output)
        if not candidate:
            indicators.append(0.0)
            continue
        try:
            is_correct = grade_answer(candidate, normalized_gold)
        except Exception:
            is_correct = False
        indicators.append(1.0 if is_correct else 0.0)
    return indicators


def _variance_reward(indicators: List[float]) -> Dict[str, float]:
    if not indicators:
        return {"reward": INVALID_QUESTION_PENALTY, "pass_rate": 0.0, "variance": 0.0}

    pass_rate = sum(indicators) / len(indicators)
    variance = pass_rate * (1 - pass_rate)
    exponent = -((variance - GAUSSIAN_TARGET) ** 2) / (2 * GAUSSIAN_VARIANCE)
    reward = math.exp(exponent)
    return {"reward": reward, "pass_rate": pass_rate, "variance": variance}


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

    scores: List[Dict[str, float]] = []
    for idx, original in enumerate(results):
        evaluated = final_results[idx] if idx < len(final_results) else {}
        question_valid = bool(original.get("question")) and bool(original.get("answer"))

        if not question_valid:
            scores.append({"overall": INVALID_QUESTION_PENALTY, "format": 0.0, "accuracy": 0.0})
            continue

        reasoner_outputs = evaluated.get("results", []) if isinstance(evaluated, dict) else []
        indicators = _reasoner_correctness(reasoner_outputs, original["answer"])
        if not indicators:
            scores.append({"overall": INVALID_QUESTION_PENALTY, "format": 1.0, "accuracy": 0.0})
            continue

        reward_stats = _variance_reward(indicators)
        overall_reward = reward_stats["reward"] if reward_stats["reward"] != INVALID_QUESTION_PENALTY else INVALID_QUESTION_PENALTY
        scores.append({
            "overall": overall_reward,
            "format": 1.0,
            "accuracy": reward_stats["pass_rate"],
        })

    dump_path = None
    debug_path = None
    dump_path, debug_path = _resolve_dump_paths()

    if dump_path:
        try:
            with open(dump_path, "a", encoding="utf-8") as f:
                for idx, score in enumerate(scores):
                    entry = {
                        "question": results[idx].get("question") if idx < len(results) else "",
                        "answer": results[idx].get("answer") if idx < len(results) else "",
                        "reasoner": final_results[idx] if idx < len(final_results) else {},
                        "reward": {
                            "overall": score.get("overall", 0.0),
                            "solver_accuracy": score.get("accuracy", 0.0),  # 0-1 correctness rate
                            "format": score.get("format", 0.0),
                        },
                        "run_id": os.getenv("RUN_ID"),
                        "timestamp": time.time(),
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[dump] Failed to dump questioner outputs: {e}")

    if debug_path and debug_entries:
        try:
            with open(debug_path, "a", encoding="utf-8") as f:
                for entry in debug_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[dump] Failed to dump questioner debug outputs: {e}")

    return scores
