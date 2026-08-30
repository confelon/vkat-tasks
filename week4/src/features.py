import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings

warnings.filterwarnings('ignore')


class TfidfExtractor:
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)


class BM25Extractor:
    def __init__(self, max_features: int = 5000, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        self.idf = None
        self.avgdl = None

    def fit_transform(self, texts):
        X = self.vectorizer.fit_transform(texts).toarray()
        self.avgdl = X.sum(axis=1).mean()
        df = (X > 0).sum(axis=0)
        N = X.shape[0]
        self.idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
        return self._score(X)

    def transform(self, texts):
        X = self.vectorizer.transform(texts).toarray()
        return self._score(X)

    def _score(self, X):
        dl = X.sum(axis=1, keepdims=True)
        denom = X + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
        return self.idf * (X * (self.k1 + 1)) / denom


class FastTextExtractor:
    """FastText эмбеддинги через gensim."""
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, epochs: int = 10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None

    def _tokenize(self, texts):
        return [t.lower().split() for t in texts]

    def fit_transform(self, texts):
        from gensim.models import FastText
        sentences = self._tokenize(texts)
        self.model = FastText(sentences, vector_size=self.vector_size,
                              window=self.window, min_count=self.min_count, epochs=self.epochs)
        return self.transform(texts)

    def transform(self, texts):
        sentences = self._tokenize(texts)
        embeddings = []
        for sent in sentences:
            vecs = [self.model.wv[w] for w in sent if w in self.model.wv]
            if vecs:
                embeddings.append(np.mean(vecs, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))
        return np.array(embeddings)


class BertExtractor:
    """BERT эмбеддинги через sentence-transformers."""
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

    def transform(self, texts):
        return self.model.encode(texts, show_progress_bar=True)