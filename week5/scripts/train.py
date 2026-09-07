"""Дообучить базовую модель через LoRA и сохранить адаптер.

    docker compose run --rm app python scripts/train.py

Нужен GPU. Пишет output/adapter/.
"""

from app.config import ProjectConfig
from app.dataset import DatasetPreparer
from app.model import ModelLoader
from app.training import TrainingService


def main() -> None:
    config = ProjectConfig()
    loader = ModelLoader(config)

    tokenizer = loader.load_tokenizer()
    model = loader.load_model()
    dataset = DatasetPreparer(config, tokenizer).build()

    TrainingService(config, model, tokenizer, dataset).run()


if __name__ == "__main__":
    main()
