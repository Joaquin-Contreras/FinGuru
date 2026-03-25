import hashlib
import chromadb
from chromadb.utils import embedding_functions


class VectorStore:
    """
    Thin wrapper around ChromaDB with local sentence-transformer embeddings.

    Documents are upserted so re-running the fetcher never creates duplicates.
    IDs are deterministic (MD5 of ticker + first 100 chars of text).
    """

    def __init__(self, collection_name: str = "finguru", db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)

        # Local model — downloaded once (~90 MB), then cached
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

    # ------------------------------------------------------------------ #
    #  WRITE                                                               #
    # ------------------------------------------------------------------ #

    def add_documents(self, documents: list[dict]) -> int:
        """
        Upserts documents from DataFetcher into ChromaDB.
        Returns the number of documents processed.
        """
        if not documents:
            return 0

        self.collection.upsert(
            documents=[doc["text"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents],
            ids=[self._make_id(doc) for doc in documents],
        )
        return len(documents)

    # ------------------------------------------------------------------ #
    #  READ                                                                #
    # ------------------------------------------------------------------ #

    def query(self, text: str, n_results: int = 5, ticker: str = None) -> list[dict]:
        """
        Semantic search. Optionally filter by ticker.

        Returns a list of dicts: {"text": ..., "metadata": ..., "distance": ...}
        """
        where = {"ticker": ticker.upper()} if ticker else None

        raw = self.collection.query(
            query_texts=[text],
            n_results=n_results,
            where=where,
        )

        results = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            results.append({"text": doc, "metadata": meta, "distance": round(dist, 4)})

        return results

    def count(self) -> int:
        return self.collection.count()

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #

    def _make_id(self, doc: dict) -> str:
        key = doc["metadata"]["ticker"] + doc["text"][:100]
        return hashlib.md5(key.encode()).hexdigest()
