# 1. Берём кусок текста, режем его на токены (слова и знаки препинания).
# 2. Каждому частому токену даём номер (словарь), редкие заменяем на <unk>.
# 3. Склеиваем всё в один длинный список номеров.
# 4. Dataset выдаёт окна фиксированной длины: вход x = токены [i..i+L),
#    цель y = те же токены, сдвинутые на 1 (модель учится предсказывать следующий токен).

import re

import torch
from torch.utils.data import Dataset

# слова (\w+), одиночные знаки препинания, перевод строки.
# "\n" = конец реплики.
TOKEN_RE = re.compile(r"\w+|[^\w\s]|\n")

UNK = "<unk>"  # нет в словаре


def load_text(path, max_chars=5_000_000):
    """Читает файл и возвращает первые max_chars символов.

    Обрезаем по границе диалога (пустая строка), чтобы не резать реплику посередине.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read(max_chars)
    # обрезаем до последнего разделителя диалогов
    cut = text.rfind("\n\n")
    if cut > 0:
        text = text[:cut]
    return text


def tokenize(text):
    """Текст -> список токенов. lower() чтобы словарь был меньше."""
    return TOKEN_RE.findall(text.lower())


def build_vocab(tokens, max_size=30_000):
    """max_size самых частых токенов.

    Возвращает:
      itos — список: номер -> токен (index to string)
      stoi — словарь: токен -> номер (string to index)
    """
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    most_common = sorted(counts, key=counts.get, reverse=True)[: max_size - 1]
    itos = [UNK] + most_common
    stoi = {t: i for i, t in enumerate(itos)}
    return itos, stoi


def encode(tokens, stoi):
    """Список токенов -> список номеров (незнакомые -> <unk>)."""
    unk_id = stoi[UNK]
    return [stoi.get(t, unk_id) for t in tokens]


def decode(ids, itos):
    """Список номеров -> читаемый текст (склеиваем и убираем пробелы перед пунктуацией)."""
    text = " ".join(itos[i] for i in ids)
    text = re.sub(r" ([^\w\s])", r"\1", text)  # "привет ," -> "привет,"
    text = text.replace("\n ", "\n")
    return text


class TextDataset(Dataset):
    """Режет длинный поток номеров токенов на окна длины seq_len.

    Пример при seq_len=4: ids = [5, 2, 9, 1, 7, ...]
      x = [5, 2, 9, 1]
      y = [2, 9, 1, 7]   (то же самое со сдвигом на 1 — "следующий токен")
    """

    def __init__(self, ids, seq_len):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.ids) - 1) // self.seq_len

    def __getitem__(self, i):
        start = i * self.seq_len
        x = self.ids[start : start + self.seq_len]
        y = self.ids[start + 1 : start + self.seq_len + 1]
        return x, y
