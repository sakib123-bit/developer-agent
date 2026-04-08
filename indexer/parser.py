import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass
class CodeChunk:
    """
    One unit of code that will become one vector in Pinecone.

    Think of it as a single "card" describing one function or class.
    The embedding is generated from `content` (the actual source code).
    Everything else is metadata stored alongside the vector.
    """

    # ── Identity ───────────────────────────────────────────────────────────────
    chunk_id:    str   # unique ID → md5(file_path + name)
    file_path:   str   # relative path from repo root, e.g. starlette/routing.py
    name:        str   # function or class name, e.g. "request_response"
    chunk_type:  str   # "function" | "class" | "method"
    parent_class: str  # if it's a method: the class name. else ""

    # ── Location ───────────────────────────────────────────────────────────────
    start_line:  int   # line where def/class starts
    end_line:    int   # line where it ends

    # ── Content ────────────────────────────────────────────────────────────────
    content:     str   # full source code of this chunk  ← this gets embedded
    docstring:   str   # extracted docstring (if any)
    imports:     str   # all imports from the file (context for the LLM)

    # ── Change detection ───────────────────────────────────────────────────────
    file_hash:   str   # md5 of the whole file — used to skip re-indexing


def make_chunk_id(file_path: str, name: str, parent_class: str = "") -> str:
    """
    Deterministic unique ID for a chunk.
    Same file + same function = same ID → allows upsert in Pinecone
    (update if changed, skip if same).
    """
    raw = f"{file_path}::{parent_class}::{name}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Import extractor ───────────────────────────────────────────────────────────

def extract_imports(tree: ast.Module, source_lines: list[str]) -> str:
    """
    Pull all import statements from the file as a single string.
    We attach this to every chunk so the LLM knows what's available
    in scope when reading a function.

    Example output:
        import ast
        from pathlib import Path
        from starlette.routing import Route
    """
    import_lines = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # node.lineno is 1-indexed
            line = source_lines[node.lineno - 1].strip()
            import_lines.append(line)
    return "\n".join(import_lines)


# ── Source extractor ───────────────────────────────────────────────────────────

def extract_source(node: ast.AST, source_lines: list[str]) -> tuple[str, int, int]:
    """
    Given an AST node (a function or class), return:
        (source_code, start_line, end_line)

    ast gives us line numbers but not the actual source — we get that
    by slicing the original source_lines list.
    """
    start = node.lineno - 1          # ast is 1-indexed, lists are 0-indexed
    end   = node.end_lineno          # end_lineno is inclusive and 1-indexed
    source = "\n".join(source_lines[start:end])
    return source, node.lineno, node.end_lineno


# ── Docstring extractor ────────────────────────────────────────────────────────

def extract_docstring(node: ast.AST) -> str:
    """
    Pull the docstring from a function or class node.
    Returns empty string if there's no docstring.
    """
    return ast.get_docstring(node) or ""


# ── Main parser ────────────────────────────────────────────────────────────────

def parse_file(file_path: Path, repo_root: Path) -> list[CodeChunk]:
    """
    Parse a single Python file and return a list of CodeChunks.

    Steps:
      1. Read the file
      2. Parse it into an AST
      3. Walk the AST looking for functions and classes
      4. For each one, extract source + metadata → CodeChunk

    Args:
        file_path : absolute path to the .py file
        repo_root : absolute path to the repo root (for making relative paths)

    Returns:
        List of CodeChunk — one per function/class/method found
    """
    try:
        source = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  [parser] Could not read {file_path}: {e}")
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"  [parser] Syntax error in {file_path}: {e}")
        return []

    source_lines  = source.splitlines()
    rel_path      = str(file_path.relative_to(repo_root))   # e.g. starlette/routing.py
    file_hash     = hashlib.md5(source.encode()).hexdigest()
    imports_block = extract_imports(tree, source_lines)

    chunks: list[CodeChunk] = []

    # Walk only the TOP level of the module (direct children of the module node)
    # so we don't double-count nested functions inside methods etc.
    for node in ast.iter_child_nodes(tree):

        # ── Top-level function ─────────────────────────────────────────────────
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            src, start, end = extract_source(node, source_lines)
            chunks.append(CodeChunk(
                chunk_id     = make_chunk_id(rel_path, node.name),
                file_path    = rel_path,
                name         = node.name,
                chunk_type   = "function",
                parent_class = "",
                start_line   = start,
                end_line     = end,
                content      = src,
                docstring    = extract_docstring(node),
                imports      = imports_block,
                file_hash    = file_hash,
            ))

        # ── Top-level class ────────────────────────────────────────────────────
        elif isinstance(node, ast.ClassDef):

            # One chunk for the class itself
            src, start, end = extract_source(node, source_lines)
            chunks.append(CodeChunk(
                chunk_id     = make_chunk_id(rel_path, node.name),
                file_path    = rel_path,
                name         = node.name,
                chunk_type   = "class",
                parent_class = "",
                start_line   = start,
                end_line     = end,
                content      = src,
                docstring    = extract_docstring(node),
                imports      = imports_block,
                file_hash    = file_hash,
            ))

            # One chunk per method inside the class
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_src, m_start, m_end = extract_source(child, source_lines)
                    chunks.append(CodeChunk(
                        chunk_id     = make_chunk_id(rel_path, child.name, node.name),
                        file_path    = rel_path,
                        name         = child.name,
                        chunk_type   = "method",
                        parent_class = node.name,
                        start_line   = m_start,
                        end_line     = m_end,
                        content      = method_src,
                        docstring    = extract_docstring(child),
                        imports      = imports_block,
                        file_hash    = file_hash,
                    ))

    return chunks


# ── Parse a list of files ──────────────────────────────────────────────────────

def parse_files(file_paths: list[Path], repo_root: Path) -> list[CodeChunk]:
    """
    Parse multiple files and return all chunks combined.
    This is what the embedder calls — it passes the changed file list
    from cloner.py straight into here.
    """
    all_chunks: list[CodeChunk] = []

    for file_path in file_paths:
        chunks = parse_file(file_path, repo_root)
        all_chunks.extend(chunks)
        print(f"  [parser] {file_path.relative_to(repo_root)} → {len(chunks)} chunks")

    return all_chunks


# ── Entry point (run directly to test) ────────────────────────────────────────

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from indexer.cloner import clone_or_pull, get_changed_files

    load_dotenv()

    REPO_LOCAL_PATH = Path(os.getenv("REPO_LOCAL_PATH", "./repos/starlette")).resolve()

    print("[parser] Cloning / pulling repo...")
    repo = clone_or_pull()

    print("[parser] Getting changed files...")
    changed_files = get_changed_files(repo)

    if not changed_files:
        print("[parser] Nothing changed. Exiting.")
        exit(0)

    print(f"\n[parser] Parsing {len(changed_files)} file(s)...\n")
    chunks = parse_files(changed_files, REPO_LOCAL_PATH)

    # ── Summary ────────────────────────────────────────────────────────────────
    functions = [c for c in chunks if c.chunk_type == "function"]
    classes   = [c for c in chunks if c.chunk_type == "class"]
    methods   = [c for c in chunks if c.chunk_type == "method"]

    print(f"""
── Parse Summary ──────────────────────────────
  Total chunks : {len(chunks)}
  Functions    : {len(functions)}
  Classes      : {len(classes)}
  Methods      : {len(methods)}
""")

    # Show a few examples so you can see what a chunk looks like
    print("── Sample chunks ──────────────────────────────")
    for chunk in chunks[:3]:
        print(f"""
  name        : {chunk.name}
  type        : {chunk.chunk_type}
  parent      : {chunk.parent_class or '—'}
  file        : {chunk.file_path}
  lines       : {chunk.start_line} → {chunk.end_line}
  docstring   : {chunk.docstring[:80] + '...' if len(chunk.docstring) > 80 else chunk.docstring or '—'}
  chunk_id    : {chunk.chunk_id}
  content preview:
{chr(10).join('    ' + l for l in chunk.content.splitlines()[:6])}
    ...
""")