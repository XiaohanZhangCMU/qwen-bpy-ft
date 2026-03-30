"""
Model inference for evaluation.

Supports three backends:
  - "hf"     : HuggingFace transformers (loads adapter via PEFT if checkpoint_dir set)
  - "vllm"   : vLLM server (calls OpenAI-compatible /v1/chat/completions)
  - "openai" : OpenAI API directly (for GPT-4o baseline)
"""

from __future__ import annotations

import os
from typing import Optional

from data_collection.prompt_templates import SYSTEM_PROMPT
from shared.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Inferencer:
    def generate(self, prompt: str, n: int = 1) -> list[str]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------

class HFInferencer(Inferencer):
    """
    Loads a base model + optional LoRA adapter via transformers + PEFT.
    """

    def __init__(
        self,
        model_name_or_path: str,
        checkpoint_dir: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading tokenizer", extra={"model": model_name_or_path})
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        logger.info("Loading model", extra={"model": model_name_or_path})
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if checkpoint_dir and checkpoint_dir != model_name_or_path:
            from peft import PeftModel
            logger.info("Loading LoRA adapter", extra={"adapter": checkpoint_dir})
            self.model = PeftModel.from_pretrained(self.model, checkpoint_dir)
            self.model = self.model.merge_and_unload()

        self.model.eval()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, n: int = 1) -> list[str]:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        results: list[str] = []
        for _ in range(n):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            text_out = self.tokenizer.decode(generated, skip_special_tokens=True)
            results.append(text_out)
        return results


# ---------------------------------------------------------------------------
# vLLM backend (OpenAI-compatible API)
# ---------------------------------------------------------------------------

class VLLMInferencer(Inferencer):
    """
    Calls a running vLLM server via the OpenAI-compatible API.
    Set VLLM_API_BASE (default: http://localhost:8000/v1) and optionally VLLM_API_KEY.
    """

    def __init__(
        self,
        model_id: str,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        from openai import OpenAI

        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            base_url=api_base or os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1"),
        )

    def generate(self, prompt: str, n: int = 1) -> list[str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=n,
        )
        return [choice.message.content or "" for choice in response.choices]


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class OpenAIInferencer(Inferencer):
    """
    Calls the OpenAI API directly.  Used for the GPT-4o oracle baseline.
    Reads OPENAI_API_KEY from the environment.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        from openai import OpenAI

        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, prompt: str, n: int = 1) -> list[str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=n,
        )
        return [choice.message.content or "" for choice in response.choices]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_inferencer(
    backend: str,
    model_name_or_path: str,
    checkpoint_dir: Optional[str] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 2048,
) -> Inferencer:
    if backend == "openai":
        return OpenAIInferencer(
            model_id=model_name_or_path,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
    if backend == "vllm":
        return VLLMInferencer(
            model_id=checkpoint_dir or model_name_or_path,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
    return HFInferencer(
        model_name_or_path=model_name_or_path,
        checkpoint_dir=checkpoint_dir,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
