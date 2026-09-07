"""Отправить один chat-запрос серверу vLLM через его OpenAI-совместимый API.

    docker compose up -d vllm
    docker compose run --rm app python scripts/client.py
    docker compose run --rm app python scripts/client.py --prompt "Explain what Docker is."
"""

import argparse

from openai import OpenAI

DEFAULT_PROMPT = "Give three tips for staying healthy."
# "vllm" — имя сервиса, Compose резолвит его в своей сети. С хоста тот же
# сервер доступен как http://localhost:8000/v1.
BASE_URL = "http://vllm:8000/v1"
# Имя, под которым сервер публикует модель (--served-model-name в compose).
MODEL_NAME = "alpaca-lora"


def main() -> None:
    parser = argparse.ArgumentParser(description="Задать развёрнутой модели один вопрос.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    # vLLM говорит на OpenAI API, официальный клиент работает без изменений.
    # Он требует api_key; сервер на него не смотрит.
    client = OpenAI(base_url=BASE_URL, api_key="not-needed")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=128,
        # Жадное декодирование, как в локальном скрипте проверки.
        temperature=0,
        # Страховка: нормально модель останавливается сама на EOS.
        stop=["###"],
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
