"""Подготовка данных Титаника для KAN.

Весь скучный код (загрузка, признаки, разбиение, тензоры) живёт здесь,
чтобы ноутбук остался коротким и читаемым.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

SEED = 42


def load_features(csv_path):
    """Читает CSV и строит все признаки, которые будем пробовать.

    KAN на входе хочет осмысленные числа — такие, у которых
    "больше/меньше" имеет смысл. Поэтому:

    - Age        — возраст как есть; пропуски оставляем NaN, заполним их
                   ПОСЛЕ разбиения на train/test (иначе утечка)
    - LogFare    — log1p(Fare): цена билета скошена (0..512, большинство < 50),
                   логарифм сжимает длинный хвост в компактный диапазон
    - Pclass     — класс каюты 1/2/3 как есть: порядок осмыслен (1-й лучше 3-го)
    - Sex        — 0 = мужчина, 1 = женщина
    - FamilySize — SibSp + Parch + 1 (сам пассажир)
    - IsAlone    — 1, если пассажир путешествовал один
    """
    df = pd.read_csv(csv_path)

    X = pd.DataFrame()
    X["Age"] = df["Age"]
    X["LogFare"] = np.log1p(df["Fare"])
    X["Pclass"] = df["Pclass"]
    X["Sex"] = (df["Sex"] == "female").astype(int)
    X["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

    y = df["Survived"]
    return X, y


def make_dataset(X, y, features):
    """Отбирает столбцы `features` и собирает словарь-датасет
    в формате, который ожидает pykan: 4 тензора train/test."""
    X = X[features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Пропуски (Age) заполняем медианой, посчитанной ТОЛЬКО по train:
    # статистика из test не должна участвовать в обучении.
    med = X_train.median()
    X_train = X_train.fillna(med)
    X_test = X_test.fillna(med)

    def to_tensor(a):
        return torch.tensor(np.asarray(a), dtype=torch.float32)

    return {
        "train_input": to_tensor(X_train),
        "train_label": to_tensor(y_train).reshape(-1, 1),
        "test_input": to_tensor(X_test),
        "test_label": to_tensor(y_test).reshape(-1, 1),
    }
