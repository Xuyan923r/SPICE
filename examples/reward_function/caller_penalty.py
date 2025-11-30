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
HTTP_TIMEOUT = float(os.getenv("CALLER_HTTP_TIMEOUT", "600"))
QUESTIONER_DUMP_DIR = os.getenv("QUESTIONER_DUMP_DIR")
QUESTIONER_DUMP_FILE = os.getenv("QUESTIONER_DUMP_FILE")

def fetch(index,i):
    response = requests.get(f"http://0.0.0.0:{5000+index}/hello?name={i}", timeout=HTTP_TIMEOUT)
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
        futures = [executor.submit(fetch, i,random_names[i]) for i in range(4)]

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
    if not isinstance(answer, str):
        return ""
    return answer.strip()


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
    with open('test.json','w') as f:
        json.dump(predicts,f,indent=4)
    for predict in predicts:
        question = ""
        answer = ""

        # Prefer structured JSON from SPICE challenger
        try:
            obj = json.loads(predict)
            if isinstance(obj, dict):
                gen_phase = obj.get("generation_phase", {})
                question = gen_phase.get("question", "") or ""
                answer = gen_phase.get("answer", "") or ""
        except Exception:
            pass

        # Fallback to <question> ... </question> and \boxed{} format
        if not question or not answer:
            questions = re.findall(r"<question>(.*?)</question>", predict, re.DOTALL)
            answers = extract_boxed_content(predict)
            if questions and answers:
                try:
                    question = questions[-1].strip()
                    answer = _normalize_answer_text(answers[-1])
                except Exception:
                    question, answer = "", ""

        results.append({"question": question, "answer": _normalize_answer_text(answer)})

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
    if QUESTIONER_DUMP_DIR or QUESTIONER_DUMP_FILE:
        if QUESTIONER_DUMP_FILE:
            dump_path = QUESTIONER_DUMP_FILE
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        else:
            os.makedirs(QUESTIONER_DUMP_DIR, exist_ok=True)
            dump_path = os.path.join(QUESTIONER_DUMP_DIR, "all_results.jsonl")

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

    return scores
