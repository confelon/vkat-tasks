import os
import tempfile
import numpy as np
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings('ignore')


class NaiveBayesClassifier:
    def __init__(self):
        self.model = MultinomialNB()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class KNNClassifier:
    def __init__(self, n_neighbors: int = 5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class FastTextClassifier:
    """
    Встроенный supervised-классификатор fasttext.
    Может обучаться либо из памяти (fit), либо из готового .txt файла (fit_from_file).
    """

    def __init__(self, lr: float = 0.5, epoch: int = 25, wordNgrams: int = 2):
        self.lr = lr
        self.epoch = epoch
        self.wordNgrams = wordNgrams
        self.model = None

    def _prepare(self, texts, labels=None):
        if labels is not None:
            return [f"__label__{int(l)} {t}" for t, l in zip(texts, labels)]
        return texts

    def fit(self, texts, labels):
        """Обучение из списков (через временный файл)."""
        import fasttext
        train_data = self._prepare(texts, labels)

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False
        ) as tmp:
            tmp.write("\n".join(train_data))
            tmp_path = tmp.name

        try:
            self.model = fasttext.train_supervised(
                tmp_path,
                lr=self.lr,
                epoch=self.epoch,
                wordNgrams=self.wordNgrams
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return self

    def fit_from_file(self, path: str):
        """Обучение из готового FastText .txt файла."""
        import fasttext

        self.model = fasttext.train_supervised(
            path,
            lr=self.lr,
            epoch=self.epoch,
            wordNgrams=self.wordNgrams
        )

        return self

    def predict(self, texts):
        preds = []

        for text in texts:
            labels, _ = self.model.predict(str(text))

            if not labels:
                preds.append(0)
                continue

            label = labels[0]

            if label == "__label__spam":
                preds.append(1)
            elif label == "__label__ham":
                preds.append(0)
            else:
                raise ValueError(f"Unknown FastText label: {label}")

        return np.asarray(preds, dtype=int)

    def predict_proba(self, texts):
        result = []

        for text in texts:
            labels, probs = self.model.predict(str(text))

            if not labels:
                result.append([1.0, 0.0])
                continue

            label = labels[0]
            p = float(probs[0])

            if label == "__label__spam":
                result.append([1.0 - p, p])
            elif label == "__label__ham":
                result.append([p, 1.0 - p])
            else:
                raise ValueError(f"Unknown FastText label: {label}")

        return np.asarray(result, dtype=float)


class FNNClassifier:
    def __init__(self, hidden_layers: tuple = (128, 64), max_iter: int = 300):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class XGBoostClassifier:
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, X, y):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            eval_metric='logloss',
            random_state=42
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)