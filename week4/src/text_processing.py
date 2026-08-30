import re
import html
import json
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from spam_utils import load_raw_data, clean_spam_data


URL_PATTERN = r"(https?://\S+|www\.\S+)"
HTML_TAG_PATTERN = r"<[^>]+>"
CONTROL_CHARS_PATTERN = r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
MULTI_SPACE_PATTERN = r"\s+"


# 1. Очистка текста
def remove_html_tags(text: str) -> str:
    """
    Удаляет HTML-теги и разворачивает простые HTML-сущности. Ex: '<b>Win</b> &amp; cash' -> 'Win & cash'
    """
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = re.sub(HTML_TAG_PATTERN, " ", text)
    return text


def replace_urls(text: str, token: str = "URL") -> str:
    """
    Заменяет ссылки на служебный токен.
    """
    if not isinstance(text, str):
        return ""

    return re.sub(URL_PATTERN, token, text)


def normalize_whitespace(text: str) -> str:
    """
    Удаляет управляющие символы, неразрывные пробелы и схлопывает несколько пробелов в один.
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", " ")

    text = re.sub(CONTROL_CHARS_PATTERN, " ", text)
    text = re.sub(MULTI_SPACE_PATTERN, " ", text)

    return text.strip()


def remove_special_chars(text: str) -> str:
    """
    Аккуратно убирает лишние специальные символы, но не вычищает всё подряд.
    Оставляем:
    - буквы/цифры через \w
    - пробелы
    - базовую пунктуацию: . , ! ? ; : ' -
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("_", " ")

    # Оставляем буквы/цифры, пробелы и минимальную пунктуацию
    text = re.sub(r"[^\w\s.,!?;:'-]", " ", text)

    return text


def remove_obvious_noise(text: str) -> str:
    """
    Удаляет очевидный мусор:
    - !!! -> !
    - ??? -> ?
    - ... -> убираем
    - --- -> убираем
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"!{3,}", " ! ", text)
    text = re.sub(r"\?{3,}", " ? ", text)
    text = re.sub(r"\.{3,}", " ", text)
    text = re.sub(r"-{3,}", " ", text)

    # На всякий случай еще раз подчистим пробелы
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def clean_text(text: str, url_mode: str = "token") -> str:
    """
    Основная функция очистки сообщения.

    Параметры:
        text: исходное сообщение
        url_mode:
            - 'token'  -> заменить ссылки на URL
            - 'remove' -> удалить ссылки
            - 'keep'   -> оставить ссылки как есть
    """
    text = remove_html_tags(text)

    if url_mode == "token":
        text = replace_urls(text, token=" URL ")
    elif url_mode == "remove":
        text = replace_urls(text, token=" ")
    elif url_mode == "keep":
        pass
    else:
        raise ValueError("url_mode must be one of: token, remove, keep")

    text = normalize_whitespace(text)
    text = remove_special_chars(text)
    text = remove_obvious_noise(text)
    text = normalize_whitespace(text)

    return text.strip()


def clean_series(series: pd.Series, url_mode: str = "token") -> pd.Series:
    """
    Применяет очистку к pandas Series.
    """
    return series.fillna("").astype(str).apply(
        lambda x: clean_text(x, url_mode=url_mode)
    )


# 2. Вспомогательные функции
def simple_tokenize(text: str):
    """
    Простая токенизация для BM25 / KNN / FastText.
    Специально не делаем лемматизацию и стемминг.
    """
    return re.findall(r"[a-z0-9']+", str(text).lower())


# 3. Загрузка исходного датасета
def load_dataset() -> pd.DataFrame:
    """
    Загружает CSV и приводит колонки к виду:
        label        -> ham / spam
        message      -> исходный текст

    Дубликаты и пустые Unnamed-колонки удаляются
    """
    df = load_raw_data()

    return clean_spam_data(df)


# 4. Stratified split
def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 424
):
    """
    Делает стратифицированное разбиение, чтобы доля spam/ham сохранилась в train и test.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


# 5. Подготовка файлов для FastText
def write_fasttext_file(df: pd.DataFrame, path: Path):
    """
    Формат для FastText:
        __label__spam text here
        __label__ham text here
    """
    with open(path, "w", encoding="utf-8") as f:
        for label, message in zip(df["label"], df["message_clean"]):
            message = str(message).replace("\n", " ").replace("\r", " ").strip()
            f.write(f"__label__{label} {message}\n")


# 6. Основной пайплайн
def save_stats(path: Path, stats: dict):
    path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Clean spam/ham text and create stratified train/test split."
    )

    parser.add_argument(
        "--outdir",
        default="../data/split",
        help="Output directory for prepared data"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--url-mode",
        choices=["token", "remove", "keep"],
        default="token",
        help="What to do with URLs: replace by URL token, remove, or keep"
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Загружаем сырые данные
    df_raw = load_dataset()
    raw_rows = len(df_raw)

    print("=== Raw data preview ===")
    print(df_raw.head())
    print()

    # 2. Очищаем текст
    df_raw["message_clean"] = clean_series(
        df_raw["message"],
        url_mode=args.url_mode
    )

    # 3. Проверяем пустые сообщения после очистки
    empty_mask = df_raw["message_clean"].str.strip() == ""
    empty_count = int(empty_mask.sum())

    df = df_raw[~empty_mask].copy()

    print("=== Cleaning stats ===")
    print(f"Empty messages after cleaning: {empty_count}")

    # 4. Удаляем дубликаты после очистки
    before_dedup = len(df)

    df = df.drop_duplicates(
        subset=["label", "message_clean"]
    ).reset_index(drop=True)

    duplicates_removed = before_dedup - len(df)

    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Rows after cleaning: {len(df)}")
    print()

    # 5. Stratified split 80/20
    train_df, test_df = stratified_split(
        df,
        test_size=args.test_size,
        seed=args.seed
    )

    train_path = outdir / "train.csv"
    test_path = outdir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 6. Файлы для FastText
    write_fasttext_file(train_df, outdir / "fasttext_train.txt")
    write_fasttext_file(test_df, outdir / "fasttext_test.txt")

    # 7. Сохраняем статистику сплита
    stats = {
        "raw_rows": raw_rows,
        "empty_after_cleaning": empty_count,
        "duplicates_removed": duplicates_removed,
        "rows_after_cleaning": len(df),
        "test_size": args.test_size,
        "seed": args.seed,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "class_distribution_all": df["label"].value_counts().to_dict(),
        "class_distribution_train": train_df["label"].value_counts().to_dict(),
        "class_distribution_test": test_df["label"].value_counts().to_dict(),
        "class_share_train": train_df["label"]
            .value_counts(normalize=True)
            .round(4)
            .to_dict(),
        "class_share_test": test_df["label"]
            .value_counts(normalize=True)
            .round(4)
            .to_dict(),
    }

    save_stats(outdir / "split_stats.json", stats)

    print("=== Split stats ===")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print()

    print("Train class distribution:")
    print(train_df["label"].value_counts())
    print()

    print("Train class share:")
    print(train_df["label"].value_counts(normalize=True).round(4))
    print()

    print("Test class distribution:")
    print(test_df["label"].value_counts())
    print()

    print("Test class share:")
    print(test_df["label"].value_counts(normalize=True).round(4))
    print()

    print(f"Saved files in: {outdir.resolve()}")
    print("- train.csv")
    print("- test.csv")
    print("- fasttext_train.txt")
    print("- fasttext_test.txt")
    print("- split_stats.json")


if __name__ == "__main__":
    main()