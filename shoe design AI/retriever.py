#retriever.py


from duckduckgo_search import DDGS

def search_duckduckgo(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo and return a concatenated string of result snippets."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            if "body" in r:
                results.append(r["body"])
    return "\n\n".join(results)