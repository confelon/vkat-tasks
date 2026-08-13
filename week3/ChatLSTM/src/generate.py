# generate.py — консольный чат с обученной моделью.
#
# Запуск:  python src/generate.py
# Пишете реплику -> бот отвечает. Выход: пустая строка или Ctrl+C.
#
# Как работает autoregressive-генерация:
# 1. Историю диалога форматируем как в датасете:
#    "- реплика\n- реплика\n- " — модель "видит" разговор и начало своего ответа.
# 2. Прогоняем токены через модель, берём logits последнего шага.
# 3. Делим logits на temperature, применяем softmax -> вероятности,
#    случайно выбираем следующий токен (torch.multinomial).
# 4. Добавляем токен к последовательности и повторяем, пока модель
#    не закончит реплику ("\n") или не кончится лимит токенов.
#
# temperature < 1 — ответы более предсказуемые, > 1 — более случайные.

import argparse
import sys

import torch

from data import UNK, decode, encode, tokenize
from model import LSTMLanguageModel

CHECKPOINT_PATH = "model.pt"
HISTORY_LINES = 8  # сколько последних реплик подавать модели как контекст


def generate(model, stoi, itos, history, temperature, max_tokens, device):
    # история диалога в формате датасета + начало реплики-ответа
    text = "".join(f"- {line}\n" for line in history) + "- "
    ids = encode(tokenize(text), stoi)

    x = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = None
    newline_id = stoi.get("\n")
    out_ids = []

    with torch.no_grad():
        # сначала "скармливаем" весь контекст, чтобы LSTM накопила состояние
        logits, hidden = model(x, hidden)
        for _ in range(max_tokens):
            last = logits[0, -1] / temperature       # logits последнего шага
            last[stoi[UNK]] = float("-inf")          # служебный <unk> не выбираем
            probs = torch.softmax(last, dim=-1)      # -> вероятности
            next_id = torch.multinomial(probs, 1).item()  # случайный выбор
            if next_id == newline_id:                # конец реплики
                break
            out_ids.append(next_id)
            # дальше подаём только новый токен: контекст уже в hidden
            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, hidden = model(x, hidden)

    return decode(out_ids, itos)


def main():
    # чтобы кириллица корректно ходила через Windows-консоль
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    itos = ckpt["itos"]
    stoi = {t: i for i, t in enumerate(itos)}
    cfg = ckpt["config"]

    model = LSTMLanguageModel(len(itos), cfg["emb_dim"], cfg["hidden_dim"], cfg["num_layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    print("Чат с LSTM-ботом. Пустая строка или Ctrl+C — выход.")
    history = []  # реплики диалога по очереди: вы, бот, вы, бот...
    while True:
        try:
            user_line = input("Вы:  ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not user_line:
            break

        history.append(user_line)
        answer = generate(
            model, stoi, itos, history[-HISTORY_LINES:],
            args.temperature, args.max_tokens, device,
        )
        history.append(answer)
        print(f"Бот: {answer}")


if __name__ == "__main__":
    main()
