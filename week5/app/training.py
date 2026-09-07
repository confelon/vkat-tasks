"""Дообучает базовую модель через LoRA и сохраняет адаптер."""

import time

import torch
from datasets import Dataset
from peft import PeftModel, get_peft_model
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from app.config import ProjectConfig
from app.lora import build_lora_config


class TrainingService:
    """Оборачивает модель в LoRA-слои, запускает цикл обучения, сохраняет адаптер."""

    def __init__(
        self,
        config: ProjectConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

    def run(self) -> None:
        peft_model = self._apply_lora()
        trainer = self._build_trainer(peft_model)

        start = time.perf_counter()
        trainer.train()
        minutes = (time.perf_counter() - start) / 60

        # Сохраняется только адаптер: adapter_config.json +
        # adapter_model.safetensors, несколько мегабайт. База на диске не меняется.
        peft_model.save_pretrained(self.config.adapter_dir)

        print(f"\nAdapter saved to {self.config.adapter_dir}")
        print(f"Training time:   {minutes:.1f} min")
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    def _apply_lora(self) -> PeftModel:
        # KV-кэш ускоряет только генерацию; при обучении он не нужен.
        self.model.config.use_cache = False
        peft_model = get_peft_model(self.model, build_lora_config(self.config))
        # Печатает, как мало мы реально обучаем: "trainable: 1.08M || all: 495M".
        peft_model.print_trainable_parameters()
        return peft_model

    def _build_trainer(self, peft_model: PeftModel) -> Trainer:
        args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.grad_accum_steps,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            # Смешанная точность: вычисления в fp16, оптимизатор хранит
            # копии весов в fp32.
            fp16=True,
            # Gradient checkpointing (пересчёт активаций на backward ради
            # экономии памяти) выключен: падал на моей видеокарте, а без него
            # и так влезает.
            logging_steps=10,
            # Без промежуточных чекпоинтов: прогон короткий, адаптер сохраняем
            # сами в конце.
            save_strategy="no",
            report_to="none",
        )
        return Trainer(
            model=peft_model,
            args=args,
            train_dataset=self.dataset,
            # mlm=False — causal language modelling: labels равны input_ids,
            # модель сама сдвигает их на одну позицию. Коллатор также
            # дополняет каждый батч до его самого длинного примера.
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            processing_class=self.tokenizer,
        )
