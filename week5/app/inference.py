"""Запускает дообученную модель локально — проверить адаптер до деплоя."""

import torch
from peft import PeftModel
from transformers import PreTrainedModel

from app.config import ProjectConfig
from app.dataset import format_prompt
from app.model import ModelLoader


class InferenceService:
    """Отвечает на одну инструкцию, с обученным адаптером или без него."""

    def __init__(self, config: ProjectConfig, use_adapter: bool) -> None:
        self.config = config
        loader = ModelLoader(config)
        self.tokenizer = loader.load_tokenizer()
        self.model = self._load_model(loader, use_adapter)
        # eval() выключает поведение, нужное только при обучении (dropout).
        self.model.eval()

    def _load_model(self, loader: ModelLoader, use_adapter: bool) -> PreTrainedModel:
        model = loader.load_model()
        if use_adapter:
            # Подгружает адаптер и подвешивает его к замороженным базовым
            # весам. Та же математика, что в merged-модели, но раздельно —
            # чтобы сравнивать «до/после».
            model = PeftModel.from_pretrained(model, self.config.adapter_dir)
        return model

    def generate(self, instruction: str, max_new_tokens: int = 128) -> str:
        # Тот же сборщик промпта, что при обучении: модель видит знакомый формат.
        prompt = format_prompt(instruction, "")
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                # Жадное декодирование: один промпт — всегда один и тот же
                # ответ, иначе сравнение «до/после» теряет смысл.
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # generate() возвращает промпт + ответ; оставляем только ответ.
        generated = output[0][inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)
        # Обучение научило модель останавливаться на EOS. Если она всё же
        # начинает новую секцию — обрезаем.
        return answer.split("###")[0].strip()
