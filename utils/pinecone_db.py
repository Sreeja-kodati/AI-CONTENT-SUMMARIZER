import os
from typing import List, Dict

try:
    import pinecone
except Exception:  # pragma: no cover - optional dependency
    pinecone = None


class PineconeClient:
    def __init__(self, index_name: str, api_key: str, environment: str, dimension: int = 384):
        self.index_name = index_name
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENV")
        self.dimension = dimension
        self.enabled = bool(
            pinecone is not None and self.index_name and self.api_key and self.environment
        )
        self.index = None
        if self.enabled:
            pinecone.init(api_key=self.api_key, environment=self.environment)
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(self.index_name, dimension=self.dimension)
            self.index = pinecone.Index(self.index_name)

    def upsert_chunks(self, chunk_payload: List[Dict]):
        if not self.enabled or not self.index:
            return
        self.index.upsert(vectors=chunk_payload)

    def query_similar_chunks(self, vector: List[float], top_k: int = 5) -> List[Dict]:
        if not self.enabled or not self.index:
            return []
        result = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        return result.matches if result and hasattr(result, "matches") else []
