# verl/utils/dataset.py

import os
import random
import math
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from PIL import Image
from PIL.Image import Image as ImageObject

from scripts.prompts import get_free_form_question_challenger_prompt
from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = {}
    non_tensors = {}

    for k in features[0].keys():
        values = [f[k] for f in features]
        if isinstance(values[0], torch.Tensor):
            tensors[k] = torch.stack(values)
        else:
            non_tensors[k] = np.array(values, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[Dict[str, Any], ImageObject, str],
    min_pixels: int,
    max_pixels: int
) -> ImageObject:
    """Resize / convert image for Qwen2-VL."""
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    # Resize if too large
    if (image.width * image.height) > max_pixels:
        factor = math.sqrt(max_pixels / (image.width * image.height))
        new_w, new_h = int(image.width * factor), int(image.height * factor)
        image = image.resize((new_w, new_h))

    # Resize if too small
    if (image.width * image.height) < min_pixels:
        factor = math.sqrt(min_pixels / (image.width * image.height))
        new_w, new_h = int(image.width * factor), int(image.height * factor)
        image = image.resize((new_w, new_h))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


# ====================================================================
# RLHFDataset — Free-Form Challenger Version
# ====================================================================
class RLHFDataset(Dataset):
    """Dataset supporting only Free-Form Challenger mode."""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        context_key: Optional[str] = None,
        image_key: str = "images",
        max_prompt_length: int = 4096,
        truncation: str = "error",
        use_free_form_challenger: bool = False,
        answer_type: str = "Integer",
        max_doc_tokens: int = 4096,
        max_pixels: Optional[int] = 4194304,
        min_pixels: Optional[int] = 262144,
        filter_overlong_prompts: bool = True,
    ):
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.context_key = context_key
        self.image_key = image_key

        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.use_free_form_challenger = use_free_form_challenger
        self.answer_type = answer_type
        self.max_doc_tokens = max_doc_tokens
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.filter_overlong_prompts = filter_overlong_prompts

        # =========================
        # Load dataset
        # =========================
        if "@" in data_path:
            data_path, split = data_path.split("@")
        else:
            split = "train"

        if os.path.isdir(data_path):
            self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:
            self.dataset = load_dataset(data_path, split=split)

        if self.filter_overlong_prompts:
            self.dataset = self.dataset.filter(self._filter_overlong_prompts)

    # ====================================================================
    # safe getter: fallback to example["text"]
    # ====================================================================
    def _safe_get(self, example, key):
        if key and key in example:
            v = example[key]
            if v is not None:
                return str(v)
        # fallback for your parquet
        if "text" in example:
            return str(example["text"])
        return ""

    # ====================================================================
    # Build messages (ONLY free-form challenger logic)
    # ====================================================================
    def _build_messages(self, example: Dict[str, Any]):
        if not self.use_free_form_challenger:
            raise RuntimeError("Dataset is in free-form mode, but use_free_form_challenger=False.")

        # ------------------------------------
        # 1. Read context — robust fallback
        # ------------------------------------
        context = self._safe_get(example, self.context_key)

        # ------------------------------------
        # 2. Truncate to max_doc_tokens
        # ------------------------------------
        tokens = self.tokenizer.encode(context, add_special_tokens=False)
        if len(tokens) > self.max_doc_tokens:
            start = random.randint(0, len(tokens) - self.max_doc_tokens)
            tokens = tokens[start : start + self.max_doc_tokens]
            context = self.tokenizer.decode(tokens)

        # ------------------------------------
        # 3. Build final prompt using your template
        # ------------------------------------
        prompt = get_free_form_question_challenger_prompt(
            text=context,
            answer_type=self.answer_type,
        )

        return [
            {
                "role": "user",
                "content": prompt
            }
        ]

    # ====================================================================
    def _filter_overlong_prompts(self, example):
        messages = self._build_messages(example)
        if self.tokenizer.chat_template:
            full_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            full_prompt = messages[0]["content"]
        return len(full_prompt) <= self.max_prompt_length

    # ====================================================================
    def __len__(self):
        return len(self.dataset)

    # ====================================================================
    def __getitem__(self, index):
        example = self.dataset[index]
        messages = self._build_messages(example)

        # ------------------------------------
        # Build input sequence
        # ------------------------------------
        if self.processor is None:
            full_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            model_inputs = self.tokenizer([full_prompt], add_special_tokens=False, return_tensors="pt")
        else:
            full_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            model_inputs = self.processor([full_prompt], add_special_tokens=False, return_tensors="pt")

        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]

        # ------------------------------------
        # Position ids
        # ------------------------------------
        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=model_inputs.get("image_grid_thw"),
            )
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0)

        # ------------------------------------
        # Postprocess (pad/truncate)
        # ------------------------------------
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # ------------------------------------
        # Build output dict
        # ------------------------------------
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": self.tokenizer.encode(full_prompt, add_special_tokens=False),
            "ground_truth": "",   # questioner 不需要答案
        }
