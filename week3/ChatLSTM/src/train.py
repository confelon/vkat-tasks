# текст -> токены -> словарь -> поток номеров -> train/val split
#           -> DataLoader -> обучение LSTM -> сохранение checkpoint (model.pt).

import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import TextDataset, build_vocab, encode, load_text, tokenize
from model import LSTMLanguageModel

DATA_PATH = "dialogues.txt"
CHECKPOINT_PATH = "model.pt"
MAX_CHARS = 5_000_000
VOCAB_SIZE = 30_000
SEQ_LEN = 64            # длина окна в токенах
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
EMB_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2


def evaluate(model, loader, loss_fn, device):
    """Считает средний loss на валидации (без обновления весов)."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            # CrossEntropyLoss хочет форму (N, vocab): "разворачиваем" batch и seq_len
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total += loss.item()
            count += 1
    model.train()
    return total / count


def main():
    # чтобы кириллица печаталась в Windows-консоли независимо от её кодировки
    sys.stdout.reconfigure(encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")

    # --- данные ---
    text = load_text(DATA_PATH, MAX_CHARS)
    tokens = tokenize(text)
    itos, stoi = build_vocab(tokens, VOCAB_SIZE)
    ids = encode(tokens, stoi)
    print(f"Символов: {len(text):,}, токенов: {len(ids):,}, словарь: {len(itos):,}")

    # train/val split: первые 90% потока — обучение, последние 10% — валидация
    split = int(len(ids) * 0.9)
    train_ds = TextDataset(ids[:split], SEQ_LEN)
    val_ds = TextDataset(ids[split:], SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    print(f"Батчей: train {len(train_loader)}, val {len(val_loader)}")

    # --- модель ---
    model = LSTMLanguageModel(len(itos), EMB_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- обучение ---
    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        total = 0.0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            # обрезка градиентов — стандартная защита LSTM от "взрыва" градиентов
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += loss.item()
            if step % 100 == 0:
                print(f"  эпоха {epoch}, шаг {step}/{len(train_loader)}, loss {total / step:.3f}")

        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(
            f"Эпоха {epoch}: train loss {total / len(train_loader):.3f}, "
            f"val loss {val_loss:.3f}, время {time.time() - start:.0f} c"
        )

        # сохраняем модель, если валидация улучшилась
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "itos": itos,  # словарь сохраняем вместе с весами
                    "config": {
                        "emb_dim": EMB_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "num_layers": NUM_LAYERS,
                    },
                },
                CHECKPOINT_PATH,
            )
            print(f"  сохранено в {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
