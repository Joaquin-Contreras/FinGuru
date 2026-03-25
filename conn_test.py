from data_fetcher import DataFetcher
from vector_store import VectorStore

fetcher = DataFetcher()
store = VectorStore()

# 1. Fetch y guardar en ChromaDB
print("Fetching AAPL documents...")
docs = fetcher.get_documents("AAPL", days_back=7)
n = store.add_documents(docs)
print(f"  Stored {n} documents. Total in DB: {store.count()}\n")

# 2. Query semántico
queries = [
    "analyst upgrades or downgrades",
    "earnings revenue financial results",
    "what does Apple do as a company",
]

for q in queries:
    print(f'Query: "{q}"')
    results = store.query(q, ticker="AAPL", n_results=2)
    for r in results:
        print(f"  [{r['metadata']['content_type']}] dist={r['distance']} | {r['text'][:90]}")
    print()
