from pathlib import Path


class VectorDB:
    def __init__(self, db_path: str):
        import chromadb
        path = Path(db_path).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))

    def upsert(
        self,
        collection: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        col = self.client.get_or_create_collection(name=collection)
        col.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def query(
        self,
        collection: str,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict]:
        try:
            col = self.client.get_collection(name=collection)
        except Exception:
            return []
        count = col.count()
        if count == 0:
            return []
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
        )
        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return chunks

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.list_collections()]

    def delete_collection(self, name: str) -> None:
        self.client.delete_collection(name)

    def collection_count(self, name: str) -> int:
        try:
            return self.client.get_collection(name).count()
        except Exception:
            return 0
