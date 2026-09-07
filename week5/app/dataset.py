"""Превращает сырой датасет Alpaca в токенизированные обучающие примеры.

Instruction tuning: показываем модели много пар «вопрос -> ответ» в одном
фиксированном формате, чтобы потом она узнавала этот формат и дописывала его.
Сам формат — шаблон промпта ниже, единственное место в проекте, где он задан.
"""

from datasets import Dataset, load_dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase

from app.config import ProjectConfig


def format_prompt(instruction: str, input_text: str) -> str:
    """Собирает ту часть примера, которую модель должна *прочитать*.

    Завершающее "### Response:\n" — сигнал «твоя очередь». Обучение и инференс
    должны давать байт в байт одинаковый промпт, поэтому оба идут через эту
    функцию.
    """
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


class DatasetPreparer:
    """Загружает датасет, форматирует и токенизирует."""

    def __init__(self, config: ProjectConfig, tokenizer: PreTrainedTokenizerBase) -> None:
        self.config = config
        self.tokenizer = tokenizer

    def build(self) -> Dataset:
        """Весь конвейер: сырые строки -> готовые к обучению id токенов."""
        return self.tokenize(self.load_texts())

    def load_texts(self) -> Dataset:
        """Скачать датасет, обрезать до нужного размера, отрендерить строки в текст."""
        raw = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        raw = raw.select(range(self.config.max_samples))
        return raw.map(self._to_text, remove_columns=raw.column_names)

    def tokenize(self, texts: Dataset) -> Dataset:
        """Текст -> id токенов. Без паддинга: коллатор дополнит каждый батч до
        длины самого длинного примера в нём, это экономнее, чем дополнять всё
        до max_seq_len."""
        return texts.map(self._tokenize, batched=True, remove_columns=["text"])

    def _to_text(self, example: dict[str, str]) -> dict[str, str]:
        """Одна строка датасета -> одна обучающая строка.

        Ответ приклеивается сразу к промпту, в конец добавляется токен конца
        последовательности (EOS): так модель учится, где ответ *заканчивается*.
        Без него она бы говорила до упора в лимит токенов.
        """
        prompt = format_prompt(example["instruction"], example["input"])
        return {"text": prompt + example["output"] + self.tokenizer.eos_token}

    def _tokenize(self, batch: dict[str, list[str]]) -> BatchEncoding:
        return self.tokenizer(
            batch["text"],
            truncation=True,
            max_length=self.config.max_seq_len,
        )
