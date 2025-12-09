from typing import List, Tuple

import faiss
import numpy as np

from .embeddings import get_embedding_model
from .types import Document


class Retriever:
    """
    Class to retrieve documents based on a query
    """
    def __init__(self, index: faiss.Index, documents: List[Document]):
        self.index = index
        self.documents = documents
        self.embedding_model = get_embedding_model()

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Returns top-k documents for the query
        """
        query_embedding = np.asarray(
            self.embedding_model.encode([query]), dtype="float32"
        )
        faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_embedding, k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            doc = self.documents[int(idx)]
            results.append((doc, float(score)))

        return results