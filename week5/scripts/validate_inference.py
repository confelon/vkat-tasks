"""Ответить на одну инструкцию локально — убедиться, что адаптер грузится и работает.

    docker compose run --rm app python scripts/validate_inference.py
    docker compose run --rm app python scripts/validate_inference.py --prompt "Explain what Docker is."
    docker compose run --rm app python scripts/validate_inference.py --no-adapter

Нужен GPU и output/adapter/. Запуск с --no-adapter отвечает чистой базовой
моделью — так видно, что изменило дообучение.
"""

import argparse

from app.config import ProjectConfig
from app.inference import InferenceService

DEFAULT_PROMPT = "Give three tips for staying healthy."


def main() -> None:
    parser = argparse.ArgumentParser(description="Запустить дообученную модель на одном промпте.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="ответить базовой моделью, для сравнения",
    )
    args = parser.parse_args()

    service = InferenceService(ProjectConfig(), use_adapter=not args.no_adapter)
    answer = service.generate(args.prompt)

    label = "base model" if args.no_adapter else "base model + LoRA adapter"
    print(f"\n--- {label} ---")
    print(f"### Instruction:\n{args.prompt}\n")
    print(f"### Response:\n{answer}")


if __name__ == "__main__":
    main()
