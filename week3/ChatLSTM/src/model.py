# model.py — маленькая LSTM-языковая модель.
#
# Архитектура: Embedding -> LSTM -> Linear -> logits по словарю.
# - Embedding превращает номер токена в вектор (обучаемая "таблица смыслов").
# - LSTM читает последовательность векторов и держит "память" о контексте.
# - Linear превращает выход LSTM в оценки (logits) для каждого слова словаря:
#   чем больше logit, тем вероятнее это слово будет следующим.

import torch.nn as nn


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=512, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # тензоры формы (batch, seq_len, ...)
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len) — номера токенов
        emb = self.embedding(x)              # (batch, seq_len, emb_dim)
        out, hidden = self.lstm(emb, hidden) # (batch, seq_len, hidden_dim)
        logits = self.fc(out)                # (batch, seq_len, vocab_size)
        return logits, hidden
