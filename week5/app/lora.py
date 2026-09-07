"""Единственное место, где заданы гиперпараметры LoRA."""

from peft import LoraConfig, TaskType

from app.config import ProjectConfig


def build_lora_config(config: ProjectConfig) -> LoraConfig:
    """Описывает низкоранговые адаптеры, которые PEFT встроит в модель.

    LoRA замораживает все предобученные веса и вместо этого учит пару
    маленьких матриц рядом с каждым выбранным слоем: A размера (r x in) и
    B размера (out x r). Слой считает W @ x + (alpha / r) * B @ A @ x.
    Градиенты получают только A и B — поэтому при ранге 8 из сотен миллионов
    обучаемых параметров остаётся около миллиона.
    """
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        # Без дополнительных bias: адаптер меньше, а слияние с базовыми
        # весами остаётся точной операцией.
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.lora_target_modules,
    )
