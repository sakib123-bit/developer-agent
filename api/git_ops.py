"""
api/git_ops.py

Apply a unified diff to a local repo, commit it on a new branch,
push to origin, and open a GitHub PR.
"""
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)


# ── Diff parser / file applier ─────────────────────────────────────────────────

def _apply_patch(diff: str, repo_path: Path) -> list[str]:
    """
    Parse a unified diff and apply it directly to files using Python I/O.

    Why not git apply?
    git apply rejects patches for new files when the LLM writes '--- a/file'
    instead of '--- /dev/null', and it also chokes on LLM-generated diffs that
    have wrong hunk line counts. Applying the patch in Python avoids every
    git apply format constraint.

    Strategy for multiple hunks on the same file:
    Collect all hunks first, then apply them bottom-to-top so that earlier
    hunks' line numbers remain valid after later hunks are applied.

    Returns list of relative file paths that were modified/created.
    """
    diff = diff.replace("\r\n", "\n").replace("\r", "\n")
    lines = diff.split("\n")

    # { rel_path: [(old_start, old_count, hunk_body_lines), ...] }
    file_hunks: dict[str, list[tuple[int, int, list[str]]]] = defaultdict(list)
    current_file: str | None = None
    i = 0

    while i < len(lines):
        line = lines[i]

        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            i += 1
        elif line.startswith("+++ ") and not line.startswith("+++ b/"):
            # e.g. "+++ /dev/null" or "+++ filename" without b/ prefix
            current_file = line[4:].strip().lstrip("/")
            i += 1
        elif line.startswith("---") or line.startswith("diff "):
            i += 1
        elif line.startswith("@@") and current_file:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if m:
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                i += 1
                body: list[str] = []
                while i < len(lines):
                    nxt = lines[i]
                    if nxt.startswith("@@") or nxt.startswith("---") or nxt.startswith("+++"):
                        break
                    body.append(nxt)
                    i += 1
                file_hunks[current_file].append((old_start, old_count, body))
            else:
                i += 1
        else:
            i += 1

    modified: list[str] = []

    for rel_path, hunks in file_hunks.items():
        full_path = repo_path / rel_path

        if full_path.exists():
            content = full_path.read_text(encoding="utf-8").splitlines()
        else:
            content = []
            full_path.parent.mkdir(parents=True, exist_ok=True)

        # Apply hunks from bottom to top to keep earlier line numbers valid
        for old_start, old_count, hunk_body in reversed(hunks):
            is_new_file = (old_start == 0 and old_count == 0)
            new_lines = [
                l[1:] for l in hunk_body
                if l.startswith("+") or l.startswith(" ")
            ]
            # Strip leading space/+ sigil from each
            new_lines = []
            for l in hunk_body:
                if l.startswith("+"):
                    new_lines.append(l[1:])
                elif l.startswith(" "):
                    new_lines.append(l[1:])
                # "-" lines are intentionally dropped

            if is_new_file:
                content = new_lines
            else:
                idx = max(0, old_start - 1)
                content = content[:idx] + new_lines + content[idx + old_count:]

        full_path.write_text("\n".join(content) + "\n", encoding="utf-8")
        modified.append(rel_path)

    return modified


# ── GitHub helpers ─────────────────────────────────────────────────────────────

def _parse_github_owner_repo(remote_url: str) -> str | None:
    match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", remote_url)
    return match.group(1) if match else None


# ── Public API ─────────────────────────────────────────────────────────────────

def apply_diff_and_create_pr(
    repo_path: str,
    diff: str,
    issue_key: str,
    ticket_summary: str = "",
) -> dict:
    """
    1. Create branch  fix/<issue-key>
    2. Apply the diff directly (Python I/O, no git apply)
    3. Commit
    4. Push
    5. Open a GitHub PR

    Returns: { branch, commit, pr_url, error }
    """
    path   = Path(repo_path)
    branch = f"fix/{issue_key.lower()}"

    # ── 1. Create / reset branch ───────────────────────────────────────────────
    existing = subprocess.run(
        ["git", "branch", "--list", branch],
        cwd=path, capture_output=True, text=True,
    )
    if existing.stdout.strip():
        subprocess.run(["git", "branch", "-D", branch], cwd=path, check=True)

    subprocess.run(["git", "checkout", "-b", branch], cwd=path, check=True)

    # ── 2. Apply diff ─────────────────────────────────────────────────────────
    try:
        modified = _apply_patch(diff, path)
    except Exception as e:
        subprocess.run(["git", "checkout", "-"], cwd=path, capture_output=True)
        return {"branch": branch, "commit": None, "pr_url": None,
                "error": f"Failed to apply diff: {e}"}

    if not modified:
        subprocess.run(["git", "checkout", "-"], cwd=path, capture_output=True)
        return {"branch": branch, "commit": None, "pr_url": None,
                "error": "Diff produced no file changes"}

    # ── 3. Commit ─────────────────────────────────────────────────────────────
    # Stage only the files the diff touched — not unrelated files like .last_indexed_commit
    for rel_file in modified:
        subprocess.run(["git", "add", str(rel_file)], cwd=path, check=True)
    commit_msg = f"fix({issue_key}): {ticket_summary or issue_key}"
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=path, check=True)

    short_sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=path
    ).decode().strip()

    # ── 4. Push ───────────────────────────────────────────────────────────────
    push = subprocess.run(
        ["git", "push", "-u", "origin", branch],
        cwd=path, capture_output=True, text=True,
    )
    if push.returncode != 0:
        return {"branch": branch, "commit": short_sha, "pr_url": None,
                "error": f"git push failed: {push.stderr.strip()}"}

    # ── 5. Create GitHub PR ───────────────────────────────────────────────────
    github_token = os.getenv("GITHUB_TOKEN", "")
    if not github_token:
        return {"branch": branch, "commit": short_sha, "pr_url": None,
                "error": "GITHUB_TOKEN not set — branch pushed but PR not created"}

    remote_url = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=path
    ).decode().strip()

    owner_repo = _parse_github_owner_repo(remote_url)
    if not owner_repo:
        return {"branch": branch, "commit": short_sha, "pr_url": None,
                "error": f"Remote '{remote_url}' is not a GitHub repo — branch pushed but PR not created"}

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    repo_resp      = httpx.get(f"https://api.github.com/repos/{owner_repo}", headers=headers, timeout=10)
    if repo_resp.status_code == 404:
        return {"branch": branch, "commit": short_sha, "pr_url": None,
                "error": f"GitHub repo '{owner_repo}' not found — check GITHUB_TOKEN has 'repo' scope and the remote URL is correct"}
    default_branch = repo_resp.json().get("default_branch", "main")

    pr_resp = httpx.post(
        f"https://api.github.com/repos/{owner_repo}/pulls",
        headers=headers, timeout=10,
        json={
            "title": f"[{issue_key}] {ticket_summary or issue_key}",
            "body":  f"Auto-generated by Developer-Agent for Jira ticket [{issue_key}].\n\n{ticket_summary}",
            "head":  branch,
            "base":  default_branch,
        },
    )
    pr_data = pr_resp.json()
    pr_url  = pr_data.get("html_url")

    if not pr_url:
        return {"branch": branch, "commit": short_sha, "pr_url": None,
                "error": f"GitHub PR creation failed ({pr_resp.status_code}): {pr_data.get('message', 'unknown')} — repo: {owner_repo}"}

    return {"branch": branch, "commit": short_sha, "pr_url": pr_url, "error": ""}
