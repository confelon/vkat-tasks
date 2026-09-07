"""Влить обученный LoRA-адаптер в базовую модель для раздачи через vLLM.

Обучение даёт адаптер: маленькие матрицы рядом с замороженными весами,
которые применяются на каждом forward. vLLM умеет подгружать такой адаптер
на лету (--enable-lora), но его LoRA-ядра падали на моей видеокарте, поэтому
складываем адаптер в веса один раз здесь и отдаём vLLM обычную модель.
Численно это та же модель.

    docker compose run --rm app python scripts/merge_adapter.py

Нужен GPU и output/adapter/. Пишет output/merged/, около 1 GB.
"""

from peft import PeftModel

from app.config import ProjectConfig
from app.model import ModelLoader


def main() -> None:
    config = ProjectConfig()
    loader = ModelLoader(config)

    tokenizer = loader.load_tokenizer()
    base_model = loader.load_model()

    # from_pretrained подвешивает адаптер; merge_and_unload складывает его
    # в веса и снимает обёртки PEFT — остаётся обычная модель transformers.
    peft_model = PeftModel.from_pretrained(base_model, config.adapter_dir)
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(config.merged_dir)
    # vLLM читает токенизатор из папки модели, поэтому сохраняем и его.
    tokenizer.save_pretrained(config.merged_dir)

    print(f"Merged model saved to {config.merged_dir}")


if __name__ == "__main__":
    main()
