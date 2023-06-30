import math
from typing import Callable

import faiss
import numpy as np
from typing import List, Tuple, Dict


def _default_relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # The 'correct' relevance function
    # may differ depending on a few things, including:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
    # - embedding dimensionality
    # - etc.
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


class FAISSVectorStore:

    def __init__(self,
                 embed_dim: int,
                 embed_function: Callable,
                 normalize_score_fn=_default_relevance_score_fn,
                 normalize_L2: bool = False):

        self.index = faiss.IndexFlatL2(embed_dim)
        self.embed_function = embed_function
        self.normalize_score_fn = normalize_score_fn
        self._normalize_L2 = normalize_L2
        self.itod = {}

    def add_text(self, text, id):
        embedding = self.embed_function(text)
        if self._normalize_L2:
            embedding = faiss.normalize_L2(embedding)
        self.index.add(embedding)
        self.itod[len(self.itod)] = id

    def _search(self, query: str, k: int = 4, **kwargs):
        embedding = self.embed_function(query)
        if self._normalize_L2:
            embedding = faiss.normalize_L2(embedding)
        scores, idxs = self.index.search(embedding, k)
        index_list = [(self.itod[i], score if not self.normalize_score_fn else self.normalize_score_fn(score)) for i, score in zip(idxs, scores) if i > -1]
        return index_list

    def cosine_search(self, query: str, k: int = 4, **kwargs) -> List[Tuple[int, float]]:
        result = self._search(query, k, **kwargs)

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            result = [i for i in result if i[1] > score_threshold]
        return result


if __name__ == '__main__':
    index = faiss.IndexFlatL2(768)
    emd = np.random.randn(1, 768).astype(np.float32)
    index.add(emd)
    index.search()
    print()
