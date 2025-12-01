import sys
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from verl.utils.dataset import RLHFDataset   # 你改过的版本
from scripts.prompts import get_free_form_question_challenger_prompt

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_prompt.py <dataset_path> <model_path>")
        print("Example: python test_prompt.py /data/ds/10k.parquet Qwen/Qwen2.5-7B-Instruct")
        return

    dataset_path = sys.argv[1]
    model_path = sys.argv[2]

    print("📦 Loading tokenizer:", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

    print("📂 Loading dataset:", dataset_path)

    # 自动识别 parquet / 目录 / huggingface dataset
    if os.path.isdir(dataset_path):
        dataset = load_dataset("parquet", data_dir=dataset_path, split="train")
    elif os.path.isfile(dataset_path):
        dataset = load_dataset("parquet", data_files=dataset_path, split="train")
    else:
        # huggingface dataset name
        dataset = load_dataset(dataset_path, split="train")

    print(f"📊 Dataset size: {len(dataset)}")

    # =============================
    # 构造 RLHFDataset（free-form 模式）
    # =============================
    ds = RLHFDataset(
        data_path=dataset_path,
        tokenizer=tokenizer,
        processor=None,
        prompt_key="text",          # 你训练时用的
        context_key="text",         # 关键！
        answer_key="id",
        use_free_form_challenger=True,
        answer_type="integer",
        max_doc_tokens=2048,
        max_prompt_length=16000
    )

    print("🎯 Testing sample #0 ...")
    example = dataset[0]
    print("\n=== RAW EXAMPLE ===")
    print(example)

    messages = ds._build_messages(example)
    print("\n=== MESSAGES ===")
    print(messages)

    # 使用 tokenizer 的 chat-template 拼最终 prompt
    final_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    print("\n=== FINAL PROMPT (string sent to model) ===\n")
    print(final_prompt)

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
