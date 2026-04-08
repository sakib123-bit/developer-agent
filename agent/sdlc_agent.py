import asyncio
import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.append(str(Path(__file__).parent.parent))

from indexer.searcher import search, SearchResult
from client import JiraMCPClient

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
REPO_LOCAL_PATH = Path(os.getenv("REPO_LOCAL_PATH", "./repos/starlette")).resolve()
TOP_K           = 5


def _fix_context_lines(diff: str) -> str:
    """
    Pass 1 — ensure every context line inside a hunk starts with a space.
    Pass 2 — recompute @@ hunk header line counts to match the actual body.
    Both issues cause 'corrupt patch' when git apply processes the diff.
    """
    import re as _re

    diff  = diff.replace("\r\n", "\n").replace("\r", "\n")
    lines = diff.split("\n")

    # Pass 1: fix context lines
    fixed: list[str] = []
    in_hunk = False
    for line in lines:
        if line.startswith(("---", "+++")):
            in_hunk = False
            fixed.append(line)
        elif line.startswith("@@"):
            in_hunk = True
            fixed.append(line)
        elif in_hunk:
            if line == "":
                fixed.append(" ")
            elif line[0] not in ("+", "-", " ", "\\"):
                fixed.append(" " + line)
            else:
                fixed.append(line)
        else:
            fixed.append(line)

    # Pass 2: recompute hunk header counts
    result: list[str] = []
    i = 0
    while i < len(fixed):
        line = fixed[i]
        if line.startswith("@@"):
            m = _re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)", line)
            if m:
                old_start, new_start, suffix = m.group(1), m.group(2), m.group(3)
                i += 1
                body: list[str] = []
                while i < len(fixed):
                    nxt = fixed[i]
                    if nxt.startswith("@@") or nxt.startswith("---") or nxt.startswith("+++"):
                        break
                    body.append(nxt)
                    i += 1
                old_count = sum(1 for l in body if l.startswith((" ", "-")))
                new_count = sum(1 for l in body if l.startswith((" ", "+")))
                result.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{suffix}")
                result.extend(body)
                continue
            else:
                result.append(line)
        else:
            result.append(line)
        i += 1

    return "\n".join(result)

# ── State ──────────────────────────────────────────────────────────────────────
#
# This is the single dict that flows through every node.
# Each node reads what it needs and writes its output back.
# Think of it as a shared whiteboard all nodes read and write to.

class AgentState(TypedDict):
    # Input
    issue_key:       str
    repo_path:       str         # local path to the repo (overrides REPO_LOCAL_PATH)

    # After fetch_ticket node
    ticket:          dict        # full ticket details from Jira

    # After search_code node
    chunks:          list        # list of SearchResult dicts

    # After read_files node
    files_context:   str         # full file contents joined as one string

    # After generate_diff node
    diff:            str
    explanation:     str

    # Control flow
    error:           str         # set if something goes wrong


# ── LLM ───────────────────────────────────────────────────────────────────────
# Initialised lazily inside generate_diff so the key is read at call-time,
# not at module import time (which runs before load_dotenv in some entry-points).

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model       = "gpt-4o",
        temperature = 0.1,
        api_key     = os.getenv("OPENAI_API_KEY", ""),
    )


# ── Node 1: fetch_ticket ───────────────────────────────────────────────────────

async def fetch_ticket(state: AgentState) -> AgentState:
    """
    NODE 1 — Fetch the Jira ticket via MCP server.

    Reads  : state["issue_key"]
    Writes : state["ticket"]
    """
    print(f"\n[node:fetch_ticket] Fetching {state['issue_key']} from Jira...")

    try:
        client = JiraMCPClient()
        ticket = await client.get_issue(state["issue_key"])
        print(f"[node:fetch_ticket] ✓ Got ticket: {ticket['summary']}")
        return {**state, "ticket": ticket}

    except Exception as e:
        print(f"[node:fetch_ticket] ✗ Failed: {e}")
        return {**state, "error": f"Failed to fetch ticket: {e}"}


# ── Node 2: search_code ────────────────────────────────────────────────────────

def search_code(state: AgentState) -> AgentState:
    """
    NODE 2 — Search Pinecone for relevant code chunks.

    Reads  : state["ticket"]  (uses summary + description as query)
    Writes : state["chunks"]
    """
    if state.get("error"):
        return state

    print(f"\n[node:search_code] Searching Pinecone...")

    ticket = state["ticket"]
    query  = f"{ticket.get('summary', '')}\n{ticket.get('description', '')}"

    results = search(query, top_k=TOP_K)
    print(f"[node:search_code] ✓ Found {len(results)} chunks.")

    for r in results:
        label = f"{r.parent_class}.{r.name}" if r.parent_class else r.name
        print(f"  → [{r.score}] {label}  ({r.file_path} lines {r.start_line}-{r.end_line})")

    # Convert SearchResult dataclasses to dicts for state serialisation
    chunks_as_dicts = [
        {
            "name"        : r.name,
            "chunk_type"  : r.chunk_type,
            "parent_class": r.parent_class,
            "file_path"   : r.file_path,
            "start_line"  : r.start_line,
            "end_line"    : r.end_line,
            "score"       : r.score,
            "content"     : r.content,
            "docstring"   : r.docstring,
        }
        for r in results
    ]

    return {**state, "chunks": chunks_as_dicts}


# ── Conditional edge: should_continue ─────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """
    EDGE DECISION — called after search_code.

    Looks at state and returns the name of the NEXT NODE to run.

    LangGraph uses the return value to decide which node to call next:
        "read_files"   → go to read_files node
        "no_results"   → go to no_results node
        "error"        → go to error node
    """
    if state.get("error"):
        print(f"[edge] Error detected → routing to 'handle_error'")
        return "handle_error"

    if not state.get("chunks"):
        print(f"[edge] No chunks found → routing to 'no_results'")
        return "no_results"

    print(f"[edge] {len(state['chunks'])} chunks found → routing to 'read_files'")
    return "read_files"


# ── Node 3: read_files ─────────────────────────────────────────────────────────

def read_files(state: AgentState) -> AgentState:
    """
    NODE 3 — Read full file content for each unique file in the chunks.

    Why full file and not just the chunk?
    The LLM needs surrounding context — imports, class definitions,
    other methods — to generate an accurate diff. Just the chunk alone
    is not enough.

    Reads  : state["chunks"]
    Writes : state["files_context"]
    """
    print(f"\n[node:read_files] Reading full file contents...")

    seen_files    = set()
    files_context = ""

    for chunk in state["chunks"]:
        file_path = chunk["file_path"]
        if file_path in seen_files:
            continue

        seen_files.add(file_path)
        base = Path(state["repo_path"]).resolve() if state.get("repo_path") else REPO_LOCAL_PATH
        full_path = base / file_path

        if full_path.exists():
            content = full_path.read_text(encoding="utf-8")
            files_context += f"\n=== FILE: {file_path} ===\n{content}\n=== END FILE ===\n"
            print(f"  → Read {file_path} ({len(content)} chars)")
        else:
            print(f"  → {file_path} not found on disk, skipping")

    return {**state, "files_context": files_context}


# ── Node 4: generate_diff ──────────────────────────────────────────────────────

def generate_diff(state: AgentState) -> AgentState:
    """
    NODE 4 — Call OpenAI to generate a unified diff + explanation.

    Reads  : state["ticket"], state["chunks"], state["files_context"]
    Writes : state["diff"], state["explanation"]
    """
    print(f"\n[node:generate_diff] Generating diff with GPT-4o...")

    ticket  = state["ticket"]
    chunks  = state["chunks"]

    chunk_summary = "\n".join(
        f"[score={c['score']}] {c['file_path']} → "
        f"{c['parent_class'] + '.' if c['parent_class'] else ''}{c['name']} "
        f"(lines {c['start_line']}-{c['end_line']})"
        for c in chunks
    )

    system_msg = SystemMessage(content=(
        "You are an expert Python developer. "
        "You generate precise, minimal unified diffs based on Jira tickets. "
        "Only change what the ticket asks for. Never refactor unrelated code."
    ))

    human_msg = HumanMessage(content=f"""
JIRA TICKET
-----------
Key        : {ticket.get('key', state['issue_key'])}
Summary    : {ticket['summary']}
Description: {ticket['description']}
Priority   : {ticket['priority']}
Status     : {ticket['status']}

MOST RELEVANT CODE (from semantic search)
-----------------------------------------
{chunk_summary}

FULL FILE CONTENTS
------------------
{state['files_context']}

INSTRUCTIONS
------------
1. Generate a valid unified diff in this exact format:
   --- a/path/to/file.py
   +++ b/path/to/file.py
   @@ -line,count +line,count @@
    context line (unchanged, starts with space)
   -removed line (starts with -)
   +added line (starts with +)

2. After the diff write "EXPLANATION:" on a new line, then explain:
   - What you changed and in which function/class
   - Why this change satisfies the ticket requirement
   - Any assumptions you made

3. If the ticket is unclear or no changes are needed, write an empty diff
   and explain why in the EXPLANATION section.

Generate now:
""")

    response = _get_llm().invoke([system_msg, human_msg])
    raw      = response.content.strip()

    # Strip markdown code fences the LLM sometimes wraps the diff in
    import re
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```\s*$", "", raw)
    raw = raw.strip()

    # Split diff and explanation
    if "EXPLANATION:" in raw:
        parts       = raw.split("EXPLANATION:", 1)
        diff        = parts[0].strip()
        explanation = parts[1].strip()
    else:
        diff        = raw
        explanation = "No explanation provided."

    # Fix context lines missing their leading space (LLM formatting quirk)
    diff = _fix_context_lines(diff)

    print(f"[node:generate_diff] ✓ Diff generated ({len(diff)} chars)")
    return {**state, "diff": diff, "explanation": explanation}


# ── Node: no_results ───────────────────────────────────────────────────────────

def no_results(state: AgentState) -> AgentState:
    """
    NODE — Called when Pinecone returns no relevant chunks.
    Sets a friendly explanation so the API can return something useful.
    """
    print(f"\n[node:no_results] No relevant code found.")
    return {
        **state,
        "diff"       : "",
        "explanation": (
            f"No relevant code was found in the indexed repository "
            f"for ticket {state['issue_key']}. "
            f"The ticket may refer to code that hasn't been indexed yet, "
            f"or the description may need more detail."
        ),
    }


# ── Node: handle_error ─────────────────────────────────────────────────────────

def handle_error(state: AgentState) -> AgentState:
    """
    NODE — Called when an upstream node sets state["error"].
    """
    print(f"\n[node:handle_error] Handling error: {state.get('error')}")
    return {
        **state,
        "diff"       : "",
        "explanation": f"Agent encountered an error: {state.get('error', 'Unknown error')}",
    }


# ── Build the graph ────────────────────────────────────────────────────────────

def build_graph():
    """
    Wire up all nodes and edges into a LangGraph StateGraph.

    Nodes  = the actual work (functions above)
    Edges  = the connections between nodes
    Conditional edges = LangGraph calls a function to decide which node is next
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────────────────────────────────────
    graph.add_node("fetch_ticket",  fetch_ticket)
    graph.add_node("search_code",   search_code)
    graph.add_node("read_files",    read_files)
    graph.add_node("generate_diff", generate_diff)
    graph.add_node("no_results",    no_results)
    graph.add_node("handle_error",  handle_error)

    # ── Add edges ──────────────────────────────────────────────────────────────
    # START → fetch_ticket (always)
    graph.add_edge(START, "fetch_ticket")

    # fetch_ticket → search_code (always)
    graph.add_edge("fetch_ticket", "search_code")

    # search_code → ??? (conditional — depends on what search found)
    graph.add_conditional_edges(
        "search_code",        # from this node
        should_continue,      # call this function to decide
        {                     # map return value → next node
            "read_files"  : "read_files",
            "no_results"  : "no_results",
            "handle_error": "handle_error",
        }
    )

    # read_files → generate_diff (always)
    graph.add_edge("read_files", "generate_diff")

    # generate_diff → END (always)
    graph.add_edge("generate_diff", END)

    # no_results → END
    graph.add_edge("no_results", END)

    # handle_error → END
    graph.add_edge("handle_error", END)

    return graph.compile()


# ── Public API (called by api/main.py) ────────────────────────────────────────

async def run_agent(issue_key: str, repo_path: str = "") -> dict:
    """
    Run the full LangGraph agent for a Jira ticket key.
    Returns the final state dict.
    """
    graph = build_graph()

    initial_state: AgentState = {
        "issue_key"    : issue_key.strip().upper(),
        "repo_path"    : repo_path.strip(),
        "ticket"       : {},
        "chunks"       : [],
        "files_context": "",
        "diff"         : "",
        "explanation"  : "",
        "error"        : "",
    }

    final_state = await graph.ainvoke(initial_state)
    return final_state


# ── CLI ────────────────────────────────────────────────────────────────────────

async def main():
    issue_key   = os.getenv("DEMO_ISSUE_KEY", "LL-1")
    final_state = await run_agent(issue_key)

    ticket = final_state.get("ticket", {})
    chunks = final_state.get("chunks", [])

    print(f"\n{'─'*60}")
    print(f"  Ticket : {final_state['issue_key']} — {ticket.get('summary', '')}")
    print(f"  URL    : {ticket.get('url', '')}")

    print(f"\n  Relevant chunks ({len(chunks)}):")
    for c in chunks:
        label = f"{c['parent_class']}.{c['name']}" if c['parent_class'] else c['name']
        print(f"    [{c['score']}] {label} → {c['file_path']}")

    print(f"\n  Diff:\n{final_state.get('diff', '')}")
    print(f"\n  Explanation:\n{final_state.get('explanation', '')}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    asyncio.run(main())