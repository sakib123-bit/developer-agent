import os
import json
import hashlib
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from git import Repo

load_dotenv()

REPO_URL        = os.getenv("REPO_URL", "https://github.com/encode/starlette")
REPO_LOCAL_PATH = Path(os.getenv("REPO_LOCAL_PATH", "./repos/starlette")).resolve()

# We only care about Python files for now.
# Extend this list later for JS, TS, Java etc.
SUPPORTED_EXTENSIONS = {".py"}

# This file tracks the last commit we indexed.
# Simple and effective — no DB needed for this metadata.
LAST_INDEXED_COMMIT_FILE = REPO_LOCAL_PATH / ".last_indexed_commit"


# ── Clone or pull ──────────────────────────────────────────────────────────────

def clone_or_pull() -> Repo:
    """
    If repo doesn't exist locally → clone it.
    If it already exists         → pull latest changes.

    Returns the Repo object either way.
    """
    if REPO_LOCAL_PATH.exists():
        print(f"[cloner] Repo already exists at {REPO_LOCAL_PATH}, pulling latest...")
        repo = Repo(REPO_LOCAL_PATH)
        repo.remotes.origin.pull()
        print(f"[cloner] Pull complete. HEAD is now: {repo.head.commit.hexsha[:8]}")
    else:
        print(f"[cloner] Cloning {REPO_URL} → {REPO_LOCAL_PATH} ...")
        REPO_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        repo = Repo.clone_from(REPO_URL, REPO_LOCAL_PATH)
        print(f"[cloner] Clone complete. HEAD: {repo.head.commit.hexsha[:8]}")

    return repo


# ── File hash ─────────────────────────────────────────────────────────────────

def get_file_hash(file_path: Path) -> str:
    """
    MD5 hash of a file's content.
    Used to detect if a file actually changed (content-based, not time-based).
    """
    return hashlib.md5(file_path.read_bytes()).hexdigest()


# ── Get all indexable files ────────────────────────────────────────────────────

def get_all_python_files(repo: Repo) -> list[Path]:
    """
    Walk the repo and return all files with supported extensions.
    Skips hidden dirs, __pycache__, migrations, test files etc.
    """
    SKIP_DIRS = {
        ".git", "__pycache__", ".venv", "venv", "node_modules",
        "migrations", "alembic", ".mypy_cache", ".pytest_cache",
        "dist", "build", ".eggs",
    }
    SKIP_FILE_PATTERNS = {
        "test_", "_test.py", "conftest.py", "setup.py", "setup.cfg"
    }

    files = []
    root  = Path(repo.working_dir)

    for path in root.rglob("*"):
        # Skip directories we don't care about
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        # Only supported file types
        if path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        # Skip test files (we want source code, not tests)
        if any(path.name.startswith(p) or path.name.endswith(p) for p in SKIP_FILE_PATTERNS):
            continue
        if path.is_file():
            files.append(path)

    return sorted(files)


# ── Changed file detection ─────────────────────────────────────────────────────

def get_last_indexed_commit() -> str | None:
    """Read the last commit SHA we successfully indexed."""
    if LAST_INDEXED_COMMIT_FILE.exists():
        return LAST_INDEXED_COMMIT_FILE.read_text().strip()
    return None


def save_last_indexed_commit(commit_sha: str) -> None:
    """Persist the current commit SHA so next run knows where we left off."""
    LAST_INDEXED_COMMIT_FILE.write_text(commit_sha)


def get_changed_files(repo: Repo) -> list[Path]:
    """
    Returns list of Python files that changed since the last indexed commit.

    First run  → returns ALL Python files (nothing indexed yet)
    Later runs → returns only files that changed between
                 last_indexed_commit and current HEAD

    This is the core of incremental indexing — we never re-index
    files that haven't changed.
    """
    current_commit  = repo.head.commit.hexsha
    last_commit_sha = get_last_indexed_commit()

    if last_commit_sha is None:
        # First time — index everything
        print("[cloner] No previous index found. Will index all files.")
        return get_all_python_files(repo)

    if last_commit_sha == current_commit:
        print("[cloner] No new commits since last index. Nothing to do.")
        return []

    # Git diff between last indexed commit and current HEAD
    print(f"[cloner] Detecting changes: {last_commit_sha[:8]} → {current_commit[:8]}")

    last_commit = repo.commit(last_commit_sha)
    diff        = last_commit.diff(repo.head.commit)

    changed_paths = set()
    for change in diff:
        # change.a_path = old path, change.b_path = new path
        # For renames, b_path is the new name — always use b_path
        if change.b_path:
            changed_paths.add(change.b_path)
        if change.a_path:
            changed_paths.add(change.a_path)

    # Filter to only supported files that still exist on disk
    root         = Path(repo.working_dir)
    changed_files = []

    for rel_path in changed_paths:
        full_path = root / rel_path
        if (
            full_path.exists()
            and full_path.suffix in SUPPORTED_EXTENSIONS
            and full_path.is_file()
        ):
            changed_files.append(full_path)

    print(f"[cloner] {len(changed_files)} file(s) changed since last index.")
    return sorted(changed_files)


# ── Summary ────────────────────────────────────────────────────────────────────

def get_repo_summary(repo: Repo) -> dict:
    """Quick overview of the repo state — useful for logging/debugging."""
    all_files = get_all_python_files(repo)
    return {
        "repo_url":       REPO_URL,
        "local_path":     str(REPO_LOCAL_PATH),
        "current_commit": repo.head.commit.hexsha[:8],
        "commit_message": repo.head.commit.message.strip(),
        "committed_at":   datetime.fromtimestamp(
                              repo.head.commit.committed_date
                          ).isoformat(),
        "total_py_files": len(all_files),
        "last_indexed":   get_last_indexed_commit() or "never",
    }


# ── Entry point (run directly to test) ────────────────────────────────────────

if __name__ == "__main__":
    repo    = clone_or_pull()
    summary = get_repo_summary(repo)

    print("\n── Repo Summary ──────────────────────────────")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")

    changed = get_changed_files(repo)
    print(f"\n── Files to index ({len(changed)}) ───────────────────")
    for f in changed[:10]:   # show first 10 so terminal doesn't flood
        print(f"  {f.relative_to(REPO_LOCAL_PATH)}")
    if len(changed) > 10:
        print(f"  ... and {len(changed) - 10} more")