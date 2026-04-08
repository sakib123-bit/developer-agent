"""
indexer/pipeline.py

Parameterised indexing pipeline.
Accepts a repo URL or local path, clones/pulls it, then indexes into Pinecone.
Used by the API so the user can trigger indexing from the frontend.
"""
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from git import Repo

from indexer.parser import parse_files
from indexer.cloner import get_all_python_files, SUPPORTED_EXTENSIONS
from indexer.embedder import (
    get_or_create_index,
    embed_chunks,
    build_pinecone_vectors,
    upsert_to_pinecone,
    delete_stale_chunks,
)

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

REPOS_BASE = Path(os.getenv("REPOS_BASE", "./repos")).resolve()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_url(s: str) -> bool:
    return s.startswith(("http://", "https://", "git@")) or "github.com" in s


def _local_path_for(repo: str) -> Path:
    """Derive local clone path from a URL, or resolve a local path as-is."""
    if _is_url(repo):
        name = repo.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
        return REPOS_BASE / name
    return Path(repo).expanduser().resolve()


def _last_indexed_commit(local_path: Path) -> str | None:
    marker = local_path / ".last_indexed_commit"
    return marker.read_text().strip() if marker.exists() else None


def _save_indexed_commit(local_path: Path, sha: str) -> None:
    (local_path / ".last_indexed_commit").write_text(sha)


def _get_changed_files(git_repo: Repo, local_path: Path) -> list[Path]:
    """All files on first run; only changed files on subsequent runs."""
    current = git_repo.head.commit.hexsha
    last    = _last_indexed_commit(local_path)

    if last is None:
        return get_all_python_files(git_repo)

    if last == current:
        return []

    last_commit = git_repo.commit(last)
    diff        = last_commit.diff(git_repo.head.commit)

    changed: set[str] = set()
    for change in diff:
        if change.b_path:
            changed.add(change.b_path)
        if change.a_path:
            changed.add(change.a_path)

    root   = Path(git_repo.working_dir)
    result = []
    for rel in changed:
        full = root / rel
        if full.exists() and full.suffix in SUPPORTED_EXTENSIONS and full.is_file():
            result.append(full)

    return sorted(result)


# ── Public API ─────────────────────────────────────────────────────────────────

def index_repo(repo: str) -> dict:
    """
    Clone (if URL) or use (if local path) a repo, then index into Pinecone.

    Returns:
        {
            "local_path":    str,
            "status":        "indexed" | "already_indexed",
            "files_indexed": int,
            "commit":        str (short SHA),
        }
    """
    local_path = _local_path_for(repo)

    # ── Clone or pull ──────────────────────────────────────────────────────────
    if _is_url(repo):
        if local_path.exists():
            print(f"[pipeline] Fetching {local_path} ...")
            git_repo = Repo(local_path)
            git_repo.git.fetch("origin")
            # Resolve the default remote branch (main/master) and check it out.
            # The local repo may be on a feature branch (e.g. fix/ll-1) after a
            # previous apply operation — pulling from there would fail.
            try:
                default_branch = git_repo.git.symbolic_ref(
                    "refs/remotes/origin/HEAD"
                ).split("/")[-1]
            except Exception:
                default_branch = "main"
            git_repo.git.checkout(default_branch)
            git_repo.git.reset("--hard", f"origin/{default_branch}")
            print(f"[pipeline] Reset to origin/{default_branch}.")
        else:
            print(f"[pipeline] Cloning {repo} → {local_path} ...")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            git_repo = Repo.clone_from(repo, local_path)
    else:
        if not local_path.exists():
            raise ValueError(f"Local path not found: {local_path}")
        git_repo = Repo(local_path)

    current_commit = git_repo.head.commit.hexsha

    # ── Already up to date? ────────────────────────────────────────────────────
    if _last_indexed_commit(local_path) == current_commit:
        print(f"[pipeline] Already indexed at {current_commit[:8]}. Nothing to do.")
        return {
            "local_path":    str(local_path),
            "status":        "already_indexed",
            "files_indexed": 0,
            "commit":        current_commit[:8],
        }

    # ── Find files to index ────────────────────────────────────────────────────
    changed_files = _get_changed_files(git_repo, local_path)
    if not changed_files:
        _save_indexed_commit(local_path, current_commit)
        return {
            "local_path":    str(local_path),
            "status":        "already_indexed",
            "files_indexed": 0,
            "commit":        current_commit[:8],
        }

    print(f"[pipeline] {len(changed_files)} file(s) to index.")

    # ── Parse into chunks ──────────────────────────────────────────────────────
    chunks = parse_files(changed_files, local_path)
    if not chunks:
        _save_indexed_commit(local_path, current_commit)
        return {
            "local_path":    str(local_path),
            "status":        "indexed",
            "files_indexed": 0,
            "commit":        current_commit[:8],
        }

    print(f"[pipeline] {len(chunks)} chunks to embed.")

    # ── Embed & upsert ─────────────────────────────────────────────────────────
    index          = get_or_create_index()
    chunks_by_file = defaultdict(list)
    for chunk in chunks:
        chunks_by_file[chunk.file_path].append(chunk)

    for file_path, file_chunks in chunks_by_file.items():
        delete_stale_chunks(index, file_path)
        embeddings = embed_chunks(file_chunks)
        vectors    = build_pinecone_vectors(file_chunks, embeddings)
        upsert_to_pinecone(index, vectors)
        print(f"  [pipeline] ✓ {file_path}")

    _save_indexed_commit(local_path, current_commit)
    print(f"[pipeline] Done. Commit {current_commit[:8]} saved.")

    return {
        "local_path":    str(local_path),
        "status":        "indexed",
        "files_indexed": len(changed_files),
        "commit":        current_commit[:8],
    }
