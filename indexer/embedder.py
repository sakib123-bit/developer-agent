import os
from pathlib import Path
from collections import defaultdict
import time

from google import genai
from google.genai import types
from dotenv import load_dotenv
from pinecone import Pinecone

from indexer.parser import CodeChunk, parse_files
from indexer.cloner import (
    clone_or_pull,
    get_changed_files,
    save_last_indexed_commit,
)

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sdlc-index")
REPO_LOCAL_PATH     = Path(os.getenv("REPO_LOCAL_PATH", "./repos/starlette")).resolve()

GEMINI_MODEL   = "gemini-embedding-001"
EMBEDDING_DIM  = 3072   # gemini-embedding-001 default output dimension
BATCH_SIZE     = 100    # Gemini supports up to 100 inputs per request
PINECONE_BATCH = 100    # Pinecone upsert batch size

# ── Clients ────────────────────────────────────────────────────────────────────

gemini_client   = genai.Client(api_key=GOOGLE_API_KEY, http_options={"api_version": "v1beta"})
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)


# ── Pinecone index ─────────────────────────────────────────────────────────────

def get_or_create_index():
    """
    Get the Pinecone index, creating it if it doesn't exist yet.

    ServerlessSpec means Pinecone manages the infrastructure —
    you don't provision any servers. Free tier uses this.
    """
    # existing = [idx.name for idx in pinecone_client.list_indexes()]

    # if PINECONE_INDEX_NAME not in existing:
    #     print(f"[embedder] Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
    #     pinecone_client.create_index(
    #         name      = PINECONE_INDEX_NAME,
    #         dimension = EMBEDDING_DIM,
    #         metric    = "cosine",
    #         spec      = ServerlessSpec(cloud="aws", region="us-east-1"),
    #     )
    #     # Wait for index to be ready
    #     while not pinecone_client.describe_index(PINECONE_INDEX_NAME).status["ready"]:
    #         print("[embedder] Waiting for index to be ready...")
    #         time.sleep(2)
    #     print("[embedder] Index created and ready.")
    # else:
    #     print(f"[embedder] Using existing index '{PINECONE_INDEX_NAME}'.")

    return pinecone_client.Index(PINECONE_INDEX_NAME)


# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[CodeChunk]) -> list[list[float]]:
    all_embeddings = []
    total          = len(chunks)

    for i in range(0, total, BATCH_SIZE):
        batch         = chunks[i : i + BATCH_SIZE]
        batch_texts   = [c.content for c in batch]
        batch_num     = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  [embedder] Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        while True:
            try:
                result = gemini_client.models.embed_content(
                    model    = GEMINI_MODEL,
                    contents = batch_texts,
                    config   = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                all_embeddings.extend([e.values for e in result.embeddings])
                time.sleep(1)  # small pause between successful batches
                break          # success — move to next batch
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"  [embedder] Rate limit hit. Sleeping 60s then retrying...")
                    time.sleep(60)
                    # loop continues → retries same batch
                else:
                    raise

    return all_embeddings


# ── Pinecone upsert ────────────────────────────────────────────────────────────

def build_pinecone_vectors(
    chunks: list[CodeChunk],
    embeddings: list[list[float]],
) -> list[dict]:
    """
    Pair each chunk with its embedding and build the Pinecone upsert format.

    Pinecone vector format:
    {
        "id"      : unique string ID
        "values"  : [0.023, -0.14, ...]   the embedding vector
        "metadata": { ...anything you want stored alongside... }
    }

    Metadata is what you get BACK when you search — so store everything
    useful for the agent: file path, function name, line numbers, the
    actual source code, docstring etc.

    Note: Pinecone metadata values must be str, int, float, bool, or list[str].
    """
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id"      : chunk.chunk_id,
            "values"  : embedding,
            "metadata": {
                "name"         : chunk.name,
                "file_path"    : chunk.file_path,
                "chunk_type"   : chunk.chunk_type,   # function / class / method
                "parent_class" : chunk.parent_class,
                "start_line"   : chunk.start_line,
                "end_line"     : chunk.end_line,
                "docstring"    : chunk.docstring[:500],  # Pinecone metadata size limit
                "content"      : chunk.content[:10000],  # the source code itself
                "imports"      : chunk.imports[:2000],
                "file_hash"    : chunk.file_hash,
            },
        })
    return vectors


def upsert_to_pinecone(index, vectors: list[dict]) -> None:
    """
    Upsert vectors into Pinecone in batches.

    Upsert = insert if new, update if same ID already exists.
    This is why chunk_id is deterministic (md5 of file+name) —
    re-indexing a changed function updates it, not duplicates it.
    """
    total = len(vectors)
    for i in range(0, total, PINECONE_BATCH):
        batch     = vectors[i : i + PINECONE_BATCH]
        batch_num = i // PINECONE_BATCH + 1
        total_batches = (total + PINECONE_BATCH - 1) // PINECONE_BATCH
        print(f"  [embedder] Upserting batch {batch_num}/{total_batches} ({len(batch)} vectors)...")
        index.upsert(vectors=batch)

    print(f"  [embedder] Upserted {total} vectors total.")


# ── Delete stale chunks ────────────────────────────────────────────────────────
def delete_stale_chunks(index, file_path: str) -> None:
    print(f"  [embedder] Removing old vectors for {file_path}...")
    try:
        index.delete(filter={"file_path": {"$eq": file_path}})
    except Exception as e:
        if "Namespace not found" in str(e) or "404" in str(e):
            pass  # nothing to delete yet, that's fine
        else:
            raise


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_embedder() -> None:
    """
    Full pipeline:
      1. Clone/pull repo
      2. Find changed files
      3. Parse into chunks
      4. Embed with Voyage AI
      5. Upsert into Pinecone
      6. Save commit SHA
    """
    print("\n[embedder] ── Starting indexing pipeline ──────────────────\n")

    # Step 1 — clone or pull
    repo           = clone_or_pull()
    current_commit = repo.head.commit.hexsha

    # Step 2 — find what changed
    changed_files = get_changed_files(repo)
    if not changed_files:
        print("[embedder] Nothing changed since last index. Done.")
        return

    print(f"\n[embedder] {len(changed_files)} file(s) to index.\n")

    # Step 3 — parse into chunks
    print("[embedder] Parsing files into chunks...")
    chunks = parse_files(changed_files, REPO_LOCAL_PATH)
    if not chunks:
        print("[embedder] No chunks extracted. Done.")
        return
    print(f"[embedder] Got {len(chunks)} chunks total.\n")

    # Step 4 — connect to Pinecone
    index = get_or_create_index()

    # Step 5 — for each changed file: delete old vectors, embed new, upsert
    # Group chunks by file so we can delete per file before upserting
    chunks_by_file: dict[str, list[CodeChunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_file[chunk.file_path].append(chunk)

    all_vectors = []

    for file_path, file_chunks in chunks_by_file.items():
        print(f"\n[embedder] Processing {file_path} ({len(file_chunks)} chunks)...")
        delete_stale_chunks(index, file_path)
        embeddings = embed_chunks(file_chunks)
        vectors = build_pinecone_vectors(file_chunks, embeddings)
        upsert_to_pinecone(index, vectors)   # ← upsert immediately, not at the end
        print(f"  [embedder] ✓ {file_path} saved to Pinecone.")

    # Step 6 — upsert everything
    print(f"\n[embedder] Upserting {len(all_vectors)} vectors into Pinecone...")
    upsert_to_pinecone(index, all_vectors)

    # Step 7 — save commit SHA so next run only processes new changes
    save_last_indexed_commit(current_commit)
    print(f"\n[embedder] Saved commit {current_commit[:8]} as last indexed.")

    # ── Final stats ────────────────────────────────────────────────────────────
    stats = index.describe_index_stats()
    print(f"""
[embedder] ── Done ─────────────────────────────────────────
  Vectors in Pinecone : {stats['total_vector_count']}
  Namespace           : {list(stats.get('namespaces', {}).keys()) or 'default'}
""")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_embedder()