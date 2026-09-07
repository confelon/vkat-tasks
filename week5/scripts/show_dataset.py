"""Посмотреть на обучающие данные до того, как тратить на них время GPU.

Печатает два вида примера, распределение длин в токенах и спецтокены,
которые определяют, где заканчивается ответ. Работает на CPU: скачивает
датасет и файлы токенизатора, но не веса модели.

    docker compose run --rm app python scripts/show_dataset.py
"""

from app.config import ProjectConfig
from app.dataset import DatasetPreparer
from app.model import ModelLoader


def show(title: str, body: str) -> None:
    print(f"\n=== {title} ===")
    print(body)


def main() -> None:
    config = ProjectConfig()
    tokenizer = ModelLoader(config).load_tokenizer()
    preparer = DatasetPreparer(config, tokenizer)

    texts = preparer.load_texts()
    tokenized = preparer.tokenize(texts)

    show(
        "Configuration",
        f"model      : {config.model_name}\n"
        f"dataset    : {config.dataset_name} (split {config.dataset_split})\n"
        f"samples    : {len(texts)} of the full dataset\n"
        f"max_seq_len: {config.max_seq_len} tokens",
    )

    # Два id ниже должны различаться. Иначе паддинг и токен конца ответа —
    # один символ, и Trainer молча выкинет EOS из loss.
    show(
        "Special tokens",
        f"eos_token: {tokenizer.eos_token!r} -> id {tokenizer.eos_token_id}\n"
        f"pad_token: {tokenizer.pad_token!r} -> id {tokenizer.pad_token_id}",
    )

    with_input = next(t for t in texts["text"] if "### Input:" in t)
    without_input = next(t for t in texts["text"] if "### Input:" not in t)
    show("Sample without an input block", without_input)
    show("Sample with an input block", with_input)

    lengths = [len(ids) for ids in tokenized["input_ids"]]
    truncated = sum(1 for n in lengths if n == config.max_seq_len)
    show(
        "Token lengths",
        f"min {min(lengths)} / mean {sum(lengths) // len(lengths)} / max {max(lengths)}\n"
        f"truncated at the limit: {truncated} of {len(lengths)} samples",
    )

    # Что модель реально потребляет: плоский список целых чисел.
    show("First sample as token ids (first 24)", str(tokenized["input_ids"][0][:24]))


if __name__ == "__main__":
    main()
