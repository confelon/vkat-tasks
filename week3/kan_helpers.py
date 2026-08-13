"""Хелперы для KAN-экспериментов: обучение, метрики, графики, pruning.

Ноутбук вызывает эти функции по одной строке, а весь boilerplate живёт здесь.
"""

import contextlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)

# Цвета из валидированной палитры (первые категориальные слоты + ink/grid)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

LAYOUT = dict(
    plot_bgcolor=SURFACE,
    paper_bgcolor=SURFACE,
    font=dict(color=INK),
    margin=dict(t=60, b=50, l=60, r=20),
)


def train(model, dataset, steps=30, lamb=0.0):
    """Обучает KAN как бинарный классификатор и возвращает историю ошибок.

    Оптимизатор LBFGS — стандартный выбор pykan для маленьких сетей.
    lamb > 0 включает регуляризацию, которая прижимает лишние рёбра к нулю.
    Прогресс-бар pykan прячем, чтобы не засорять вывод ячейки.
    """
    with contextlib.redirect_stderr(io.StringIO()):
        return model.fit(dataset, opt="LBFGS", steps=steps, lamb=lamb,
                         loss_fn=torch.nn.BCEWithLogitsLoss())


def probs(model, x):
    """Вероятность выживания: выход сети (логит) -> сигмоида -> число 0..1."""
    with torch.no_grad():
        logit = model(x)
    return torch.sigmoid(logit).numpy().ravel()


def metrics_table(model, dataset):
    """Метрики отдельно на train и test.

    Читать так: большой разрыв train/test — переобучение,
    одинаково слабые цифры на обеих частях — недообучение.
    """
    rows = {}
    for part in ("train", "test"):
        p = probs(model, dataset[part + "_input"])
        y = dataset[part + "_label"].numpy().ravel()
        label = (p > 0.5).astype(int)  # порог 0.5: выше — «выжил»
        rows[part] = {
            "Accuracy": accuracy_score(y, label),
            "F1": f1_score(y, label),
            "Recall": recall_score(y, label),
            "ROC-AUC": roc_auc_score(y, p),
            "LogLoss": log_loss(y, p),
        }
    return pd.DataFrame(rows).T.round(3)


def plot_training(results):
    """Кривые ошибки по шагам обучения.

    pykan хранит в истории sqrt(loss) — это задумано под RMSE-регрессию.
    У нас loss = LogLoss, поэтому возводим обратно в квадрат.
    """
    fig = go.Figure()
    for name, color in (("train", BLUE), ("test", ORANGE)):
        loss = np.asarray(results[name + "_loss"], dtype=float) ** 2
        fig.add_scatter(y=loss, name=name, mode="lines",
                        line=dict(color=color, width=2))
    fig.update_layout(
        title="Ошибка во время обучения", height=350, **LAYOUT)
    fig.update_xaxes(title="шаг обучения", gridcolor=GRID)
    fig.update_yaxes(title="LogLoss", gridcolor=GRID)
    return fig


def plot_edge_functions(model, dataset, names, layer=0):
    """Одномерные функции phi(x), которые сеть выучила на рёбрах слоя.

    У KAN обучаются не числа-веса, а маленькие функции одной переменной —
    по одной на каждое ребро «вход -> нейрон». Рисуем каждую:
    по оси X — значение признака, по оси Y — вклад ребра в сумму.
    """
    with torch.no_grad():
        model(dataset["train_input"])  # прогон, чтобы pykan сохранил активации
    x_all = model.spline_preacts[layer].numpy()   # (пример, нейрон, вход)
    y_all = model.spline_postacts[layer].numpy()

    n_out, n_in = x_all.shape[1], x_all.shape[2]
    titles = [names[i] if n_out == 1 else f"{names[i]} → n{j}"
              for j in range(n_out) for i in range(n_in)]
    fig = make_subplots(rows=n_out, cols=n_in, subplot_titles=titles)
    for j in range(n_out):
        for i in range(n_in):
            order = np.argsort(x_all[:, j, i])
            fig.add_scatter(x=x_all[order, j, i], y=y_all[order, j, i],
                            mode="lines", line=dict(color=BLUE, width=2),
                            showlegend=False, row=j + 1, col=i + 1)
    fig.update_layout(
        title="Выученные функции рёбер: вклад признака в сумму (логит)",
        height=280 * n_out + 80, **LAYOUT)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def plot_feature_scores(model, names):
    """Важность признаков по мнению самой сети (model.attribute).

    Это суммарный «размах» вклада каждого входа: признак с важностью
    около нуля сеть фактически не использует — кандидат на pruning.
    """
    model.attribute()
    scores = model.feature_score.detach().numpy()
    order = np.argsort(scores)
    fig = go.Figure(go.Bar(
        x=scores[order], y=[names[i] for i in order],
        orientation="h", marker_color=BLUE))
    fig.update_layout(title="Важность признаков (attribution)",
                      height=300, **LAYOUT)
    fig.update_xaxes(title="важность", gridcolor=GRID)
    return fig


def plot_threshold_curves(model, dataset):
    """Precision/Recall/F1 на train при разных порогах классификации.

    Возвращает график и порог с лучшим F1. Порог подбираем по train,
    а проверяем на test — выбирать его по test нельзя, это та же
    подгонка под тестовую выборку.
    """
    p = probs(model, dataset["train_input"])
    y = dataset["train_label"].numpy().ravel()
    thresholds = np.arange(0.05, 0.951, 0.01)
    curves = {
        "Precision": [precision_score(y, p > t, zero_division=0)
                      for t in thresholds],
        "Recall": [recall_score(y, p > t) for t in thresholds],
        "F1": [f1_score(y, p > t) for t in thresholds],
    }
    best = float(thresholds[int(np.argmax(curves["F1"]))])

    fig = go.Figure()
    for (name, vals), color in zip(curves.items(), (BLUE, ORANGE, AQUA)):
        fig.add_scatter(x=thresholds, y=vals, name=name, mode="lines",
                        line=dict(color=color, width=2))
    fig.add_vline(x=best, line_dash="dash", line_color=INK,
                  annotation_text=f"лучший F1 при {best:.2f}")
    fig.update_layout(title="Метрики на train в зависимости от порога",
                      height=350, **LAYOUT)
    fig.update_xaxes(title="порог", gridcolor=GRID)
    fig.update_yaxes(title="значение метрики", gridcolor=GRID)
    return fig, best


def threshold_table(model, dataset, thresholds):
    """Test-метрики при разных порогах.

    ROC-AUC и LogLoss от порога не зависят, поэтому здесь их нет,
    зато добавлен Precision — вторая половина обмена с Recall.
    """
    p = probs(model, dataset["test_input"])
    y = dataset["test_label"].numpy().ravel()
    rows = {}
    for t in thresholds:
        pred = (p > t).astype(int)
        rows[f"порог {t:.2f}"] = {
            "Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred),
            "F1": f1_score(y, pred),
        }
    return pd.DataFrame(rows).T.round(3)


def sklearn_metrics_row(clf, dataset):
    """Обучает sklearn-модель на тех же тензорах и считает те же
    test-метрики, что metrics_table — для честного сравнения с KAN."""
    clf.fit(dataset["train_input"].numpy(),
            dataset["train_label"].numpy().ravel())
    y = dataset["test_label"].numpy().ravel()
    p = clf.predict_proba(dataset["test_input"].numpy())[:, 1]
    pred = (p > 0.5).astype(int)
    return pd.Series({
        "Accuracy": accuracy_score(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "ROC-AUC": roc_auc_score(y, p),
        "LogLoss": log_loss(y, p),
    }).round(3)


def prune_features(model, names, threshold=0.03):
    """Убирает входы, которые сеть не использует (важность < threshold).

    Возвращает новую модель и список оставшихся признаков.
    Квирк pykan: после prune_input модель снова включает автосохранение
    чекпоинтов — выключаем обратно.
    """
    model.attribute()
    pruned = model.prune_input(threshold=threshold)
    pruned.auto_save = False
    kept = [names[i] for i in pruned.input_id]
    return pruned, kept
