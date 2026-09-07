"""Загружает токенизатор и базовую модель."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from app.config import ProjectConfig


class ModelLoader:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        # У Qwen2.5 pad_token == eos_token (оба <|endoftext|>). Токены
        # паддинга исключаются из loss, значит исключился бы и EOS, который мы
        # добавляем к каждому ответу, — модель не научилась бы останавливаться.
        # <|fim_pad|> — запасной спецтокен, в реальном тексте не встречается.
        tokenizer.pad_token = "<|fim_pad|>"
        # Decoder-only модель читает слева направо, паддинг — справа.
        tokenizer.padding_side = "right"
        return tokenizer

    def load_model(self) -> PreTrainedModel:
        # fp16: вдвое меньше памяти на веса (~1 GB для 0.5B) и на все
        # промежуточные тензоры. bf16 на моей видеокарте нет.
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=torch.float16,
            # "eager" — обычные matmul + softmax для attention. Дефолтный
            # "sdpa" (fused-ядра) быстрее и экономнее, но падал на моей
            # видеокарте с illegal memory access.
            attn_implementation="eager",
        )
        return model.to("cuda")
