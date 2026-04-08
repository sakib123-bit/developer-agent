import os
from dataclasses import dataclass

from google import genai
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sdlc-index")

GEMINI_MODEL = "gemini-embedding-001"
TOP_K        = 5   # how many results to return per search

gemini_client   = genai.Client(api_key=GOOGLE_API_KEY, http_options={"api_version": "v1beta"})
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# ── Result structure ───────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """
    One result from a Pinecone search.
    Contains everything the agent needs to understand the code.
    """
    score:        float   # similarity score 0-1, higher = more relevant
    name:         str     # function or class name
    chunk_type:   str     # function / class / method
    parent_class: str     # if method: which class it belongs to
    file_path:    str     # e.g. starlette/routing.py
    start_line:   int
    end_line:     int
    content:      str     # the actual source code  ← agent reads this
    docstring:    str


# ── Embed query ────────────────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """
    Convert a natural language query into a vector using Gemini.

    Note: input_type="query" vs "document" during indexing.
    Gemini trains these as a matched pair:
      - document = "represent this code fully"
      - query    = "find code relevant to this description"
    Using the right type gives better search results.
    """
    from google.genai import types
    result = gemini_client.models.embed_content(
        model    = GEMINI_MODEL,
        contents = query,
        config   = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values
    

# ── Search ─────────────────────────────────────────────────────────────────────

def search(query: str, top_k: int = TOP_K, filter: dict = None) -> list[SearchResult]:
    """
    Search Pinecone for code chunks relevant to the query.

    Args:
        query  : natural language description, e.g. from a Jira ticket
        top_k  : number of results to return (default 5)
        filter : optional Pinecone metadata filter, e.g.:
                 {"chunk_type": {"$eq": "function"}}
                 {"file_path": {"$eq": "starlette/routing.py"}}

    Returns:
        List of SearchResult sorted by relevance score (highest first)
    """
    print(f"[searcher] Embedding query...")
    query_vector = embed_query(query)

    print(f"[searcher] Searching Pinecone (top {top_k})...")
    index = pinecone_client.Index(PINECONE_INDEX_NAME)

    search_kwargs = {
        "vector"          : query_vector,
        "top_k"           : top_k,
        "include_metadata": True,   # we need metadata to get the source code back
    }
    if filter:
        search_kwargs["filter"] = filter

    response = index.query(**search_kwargs)

    # ── Parse results ──────────────────────────────────────────────────────────
    results = []
    for match in response["matches"]:
        meta = match["metadata"]
        results.append(SearchResult(
            score        = round(match["score"], 4),
            name         = meta.get("name", ""),
            chunk_type   = meta.get("chunk_type", ""),
            parent_class = meta.get("parent_class", ""),
            file_path    = meta.get("file_path", ""),
            start_line   = int(meta.get("start_line", 0)),
            end_line     = int(meta.get("end_line", 0)),
            content      = meta.get("content", ""),
            docstring    = meta.get("docstring", ""),
        ))

    return results


# ── Helpers the agent will use ─────────────────────────────────────────────────

def search_by_file(query: str, file_path: str, top_k: int = TOP_K) -> list[SearchResult]:
    """Search within a specific file only."""
    return search(query, top_k=top_k, filter={"file_path": {"$eq": file_path}})


def search_functions_only(query: str, top_k: int = TOP_K) -> list[SearchResult]:
    """Return only standalone functions, not classes or methods."""
    return search(query, top_k=top_k, filter={"chunk_type": {"$eq": "function"}})


def search_in_class(query: str, class_name: str, top_k: int = TOP_K) -> list[SearchResult]:
    """Search for methods inside a specific class."""
    return search(query, top_k=top_k, filter={"parent_class": {"$eq": class_name}})


# ── Pretty print ───────────────────────────────────────────────────────────────

def print_results(results: list[SearchResult], query: str) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Query: {query}")
    print(f"  Found: {len(results)} results")
    print(sep)

    for i, r in enumerate(results, 1):
        label = f"{r.parent_class}.{r.name}" if r.parent_class else r.name
        print(f"""
  [{i}] {label}  ({r.chunk_type})
       file  : {r.file_path}  lines {r.start_line}-{r.end_line}
       score : {r.score}
       doc   : {r.docstring[:100] + '...' if len(r.docstring) > 100 else r.docstring or '—'}

       source:
{chr(10).join('         ' + l for l in r.content.splitlines()[:8])}
       {'...' if len(r.content.splitlines()) > 8 else ''}
""")
    print(sep)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with a few different queries so you can see how well the search works

    test_queries = [
        "handle incoming HTTP request and return response",
        "websocket connection handling",
        "middleware that catches exceptions",
        "parse cookies from request headers",
    ]

    for query in test_queries:
        results = search(query)
        print_results(results, query)
        print()