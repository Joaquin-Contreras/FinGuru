from data_fetcher import DataFetcher

fetcher = DataFetcher()
docs = fetcher.get_documents("GOOGL", days_back=7)

for doc in docs:
    print(doc["metadata"]["content_type"], "->", doc["text"][:80])