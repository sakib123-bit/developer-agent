import base64
import os
import sys

import httpx
from fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv()
# ── Config ─────────────────────────────────────────────────────────────────────

JIRA_BASE_URL  = os.environ.get("JIRA_URL", "").rstrip("/")
JIRA_EMAIL     = os.environ.get("JIRA_EMAIL", "")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "")
issue_key = os.environ.get("DEMO_ISSUE_KEY", "LL-1")
if not all([JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN]):
    print(
        "ERROR: JIRA_BASE_URL, JIRA_EMAIL, and JIRA_API_TOKEN must all be set.",
        file=sys.stderr,
    )
    sys.exit(1)

_token = base64.b64encode(f"{JIRA_EMAIL}:{JIRA_API_TOKEN}".encode()).decode()
HEADERS = {
    "Authorization": f"Basic {_token}",
    "Accept": "application/json",
}
# ADD THIS temporarily after the config lines
print(f"DEBUG JIRA_URL={JIRA_BASE_URL}", file=sys.stderr)
print(f"DEBUG JIRA_EMAIL={JIRA_EMAIL}", file=sys.stderr)
print(f"DEBUG JIRA_API_TOKEN={'SET' if JIRA_API_TOKEN else 'EMPTY'}", file=sys.stderr)
# ── FastMCP app ────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="jira-mcp",
    instructions=(
        "Use jira_get_issue to fetch complete details of any Jira ticket. "
        "Pass the issue key (e.g. PROJ-123) and get back summary, description, "
        "status, assignee, comments, and all related metadata."
    ),
)

# ── ADF → plain text ───────────────────────────────────────────────────────────

def _adf_to_text(node) -> str:
    """Recursively flatten Atlassian Document Format to plain text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "\n".join(_adf_to_text(n) for n in node)
    if isinstance(node, dict):
        t        = node.get("type", "")
        children = node.get("content", [])

        if t == "text":
            return node.get("text", "")
        if t in ("paragraph", "heading"):
            return _adf_to_text(children) + "\n"
        if t == "bulletList":
            return "\n".join(f"• {_adf_to_text(li)}" for li in children)
        if t == "orderedList":
            return "\n".join(f"{i+1}. {_adf_to_text(li)}" for i, li in enumerate(children))
        if t == "listItem":
            return _adf_to_text(children).strip()
        if t == "codeBlock":
            return f"\n```\n{_adf_to_text(children)}\n```\n"
        if t == "blockquote":
            return "\n".join(f"> {l}" for l in _adf_to_text(children).splitlines())
        if t == "hardBreak":
            return "\n"
        return _adf_to_text(children)
    return ""


def _name(obj) -> str:
    return (obj or {}).get("displayName") or (obj or {}).get("name") or ""


def _parse_issue(data: dict) -> dict:
    f = data.get("fields", {})

    comments = [
        {
            "author":  _name(c.get("author")),
            "created": c.get("created", ""),
            "body":    _adf_to_text(c.get("body")).strip(),
        }
        for c in f.get("comment", {}).get("comments", [])
    ]

    subtasks = [
        {
            "key":     s["key"],
            "summary": s["fields"].get("summary", ""),
            "status":  _name(s["fields"].get("status")),
        }
        for s in f.get("subtasks", [])
    ]

    linked_issues = []
    for link in f.get("issuelinks", []):
        if "inwardIssue" in link:
            issue, direction = link["inwardIssue"], link.get("type", {}).get("inward", "")
        elif "outwardIssue" in link:
            issue, direction = link["outwardIssue"], link.get("type", {}).get("outward", "")
        else:
            continue
        linked_issues.append({
            "key":       issue["key"],
            "summary":   issue.get("fields", {}).get("summary", ""),
            "status":    _name(issue.get("fields", {}).get("status")),
            "link_type": link.get("type", {}).get("name", ""),
            "direction": direction,
        })

    sprint = None
    sprints = f.get("customfield_10020") or []
    if isinstance(sprints, list) and sprints:
        last = sprints[-1]
        sprint = last.get("name") if isinstance(last, dict) else None

    parent = None
    if f.get("parent"):
        p = f["parent"]
        parent = {
            "key":     p.get("key"),
            "summary": p.get("fields", {}).get("summary", ""),
            "status":  _name(p.get("fields", {}).get("status")),
        }

    return {
        "key":              data["key"],
        "url":              f"{JIRA_BASE_URL}/browse/{data['key']}",
        "summary":          f.get("summary", ""),
        "description":      _adf_to_text(f.get("description")).strip(),
        "status":           _name(f.get("status")),
        "resolution":       _name(f.get("resolution")),
        "priority":         _name(f.get("priority")),
        "issue_type":       _name(f.get("issuetype")),
        "reporter":         _name(f.get("reporter")),
        "assignee":         _name(f.get("assignee")) or "Unassigned",
        "created":          f.get("created", ""),
        "updated":          f.get("updated", ""),
        "due_date":         f.get("duedate") or "",
        "labels":           f.get("labels", []),
        "components":       [c["name"] for c in f.get("components", [])],
        "fix_versions":     [v["name"] for v in f.get("fixVersions", [])],
        "sprint":           sprint,
        "story_points":     f.get("story_points") or f.get("customfield_10016"),
        "time_estimate":    (f.get("timetracking") or {}).get("originalEstimate"),
        "time_spent":       (f.get("timetracking") or {}).get("timeSpent"),
        "parent":           parent,
        "subtasks":         subtasks,
        "linked_issues":    linked_issues,
        "comment_count":    len(comments),
        "comments":         comments,
    }

# ── Tool ───────────────────────────────────────────────────────────────────────

@mcp.tool()
async def jira_get_issue(issue_key: str) -> dict:
    """
    Fetch complete details of a Jira ticket by its issue key.

    Returns summary, full description (plain text), status, resolution,
    priority, issue type, reporter, assignee, created/updated/due dates,
    labels, components, fix versions, sprint, story points, time tracking,
    parent, subtasks, linked issues, and all comments.

    Args:
        issue_key: Jira issue key, e.g. PROJ-123
    """
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key.strip().upper()}"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers=HEADERS, params={"fields": "*all"})
        resp.raise_for_status()
        return _parse_issue(resp.json())


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")