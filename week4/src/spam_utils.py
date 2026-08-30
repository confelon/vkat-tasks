import os
import requests
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_raw_data() -> pd.DataFrame:
    filepath = _project_root() / "data" / "spam.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists():
        url = "https://github.com/plutus123/Spam-Ham-Detector/raw/refs/heads/main/spam.csv"
        response = requests.get(url)
        with open(filepath, "wb") as f:
            f.write(response.content)
    return pd.read_csv(filepath, encoding="latin-1")


def clean_spam_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает датасет: удаляет Unnamed столбцы, переименовывает v1/v2,
    удаляет дубликаты и сбрасывает индексы.
    """
    df_clean = df.loc[:, ~df.columns.str.contains('^Unnamed')].copy()
    
    if 'v1' in df_clean.columns and 'v2' in df_clean.columns:
        df_clean.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
        
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    return df_clean


def _project_root() -> Path:
    """Находит корень проекта независимо от рабочей директории."""
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        cwd = Path.cwd()
        for parent in [cwd] + list(cwd.parents):
            if any((parent / d).exists() for d in ['data', 'src', 'notebooks']):
                return parent
        return cwd


def load_split_data(split_dir: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает готовые train.csv и test.csv из text_processing.py.
    Возвращает (train_df, test_df) с колонками:
        label         -> 0 (ham) / 1 (spam)
        message       -> исходный текст
        message_clean -> очищенный текст
    """
    if split_dir is None:
        split_dir = _project_root() / "data" / "split"
    else:
        split_dir = Path(split_dir)

    train_df = pd.read_csv(split_dir / "train.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    train_df['label'] = train_df['label'].map({'ham': 0, 'spam': 1})
    test_df['label'] = test_df['label'].map({'ham': 0, 'spam': 1})

    return train_df, test_df