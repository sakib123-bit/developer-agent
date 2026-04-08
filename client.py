import asyncio
import json
import os
import sys
from pathlib import Path
from dotenv import dotenv_values, load_dotenv
load_dotenv()
from langchain_mcp_adapters.client import MultiServerMCPClient
_env = dotenv_values("/Users/sakib.kashi/Documents/sdlc/.env")  
SERVER_SCRIPT = Path(__file__).parent / "jira_mcp_server.py"

# ── Server config ──────────────────────────────────────────────────────────────
#
# Add more servers here as your stack grows (GitHub, Slack, etc.).
# MultiServerMCPClient merges all their tools into one flat list automatically.

MCP_SERVERS = {
    "jira": {
        "transport": "stdio",
        "command":   sys.executable,
        "args":      [str(SERVER_SCRIPT)],
        "env": {
            **os.environ,
            **_env,
            "JIRA_URL":  _env.get("JIRA_URL", ""),
            "JIRA_EMAIL":     _env.get("JIRA_EMAIL", ""),
            "JIRA_API_TOKEN": _env.get("JIRA_API_TOKEN", ""),
        },
    },
    # "github": {
    #     "transport": "stdio",
    #     "command":   sys.executable,
    #     "args":      ["github_mcp_server.py"],
    # },
}


# ── Client wrapper ─────────────────────────────────────────────────────────────

class JiraMCPClient:
    """
    Wrapper around MultiServerMCPClient for langchain-mcp-adapters >= 0.1.0.

    In 0.1.0 the async context manager was removed. Tools are fetched by
    calling `await client.get_tools()` directly — the library manages the
    underlying server connections internally per call.

    Two usage patterns:

    1. Direct call (scripts / FastAPI endpoints / LangGraph nodes):
       client = JiraMCPClient()
       issue  = await client.get_issue("PROJ-123")

    2. LangGraph agent:
       client = JiraMCPClient()
       tools  = await client.get_tools()
       agent  = create_react_agent(model, tools)
       result = await agent.ainvoke({...})
    """

    def __init__(self) -> None:
        self._multi_client = MultiServerMCPClient(MCP_SERVERS)

    # ── Tools ──────────────────────────────────────────────────────────────────

    async def get_tools(self) -> list:
        """
        Return all tools from all connected MCP servers as LangChain
        BaseTool objects. Plug directly into create_react_agent() or any
        LangChain / LangGraph construct.
        """
        return await self._multi_client.get_tools()

    # ── Direct programmatic access ─────────────────────────────────────────────

    async def get_issue(self, issue_key: str) -> dict:
        """
        Fetch a Jira ticket directly — no LLM involved.
        Useful in scripts, FastAPI routes, or LangGraph nodes
        that need raw ticket data.

        Args:
            issue_key: Jira issue key, e.g. "PROJ-123"

        Returns:
            Parsed issue dict (summary, description, status, comments, …)
        """
        tools = await self.get_tools()

        tool = next((t for t in tools if t.name == "jira_get_issue"), None)
        if tool is None:
            raise RuntimeError("jira_get_issue not found — is jira_mcp_server.py reachable?")

        raw = await tool.ainvoke({"issue_key": issue_key.strip().upper()})
        if isinstance(raw, list):
            raw = raw[0].get("text", "{}") if isinstance(raw[0], dict) else raw[0]
        # Tool returns a JSON string; parse into dict.
        return json.loads(raw) if isinstance(raw, str) else raw


# ── LangGraph agent demo ───────────────────────────────────────────────────────

async def run_agent_demo(issue_key: str) -> None:
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        print("Install langchain-anthropic and langgraph to run the agent demo.")
        return

    print(f"\n── LangGraph agent demo  (ticket: {issue_key}) ──\n")

    client = JiraMCPClient()
    tools  = await client.get_tools()
    print(f"Tools loaded: {[t.name for t in tools]}\n")

    model = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    agent = create_react_agent(model, tools)

    result = await agent.ainvoke(
        {"messages": [HumanMessage(
            f"Fetch the Jira ticket {issue_key} and give me a concise summary: "
            "what it is about, current status, who it is assigned to, and any "
            "important comments."
        )]}
    )

    print("Agent response:")
    print(result["messages"][-1].content)


# ── Direct call demo ───────────────────────────────────────────────────────────

def _pretty_print(issue: dict) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {issue['key']}  {issue['url']}")
    print(sep)
    print(f"  Summary    : {issue['summary']}")
    print(f"  Type       : {issue['issue_type']}  |  Status : {issue['status']}  |  Priority: {issue['priority']}")
    print(f"  Assignee   : {issue['assignee']}    Reporter: {issue['reporter']}")
    print(f"  Sprint     : {issue['sprint'] or '—'}")
    print(f"  Labels     : {', '.join(issue['labels']) or '—'}")
    print(f"  Created    : {issue['created']}")
    print(f"  Updated    : {issue['updated']}")

    if issue["description"]:
        print(f"\n  Description:\n")
        for line in issue["description"].splitlines():
            print(f"    {line}")

    if issue["comments"]:
        print(f"\n  Comments ({issue['comment_count']}):")
        for c in issue["comments"]:
            print(f"\n    [{c['created'][:10]}]  {c['author']}")
            for line in c["body"].strip().splitlines():
                print(f"      {line}")

    print(f"\n{sep}\n")


async def run_direct_demo(issue_key: str) -> None:
    print(f"\n── Direct call demo  (ticket: {issue_key}) ──\n")
    client = JiraMCPClient()
    issue  = await client.get_issue(issue_key)
    _pretty_print(issue)


async def main() -> None:
    issue_key = os.environ.get("DEMO_ISSUE_KEY", "LL-1")
    mode      = os.environ.get("DEMO_MODE", "direct")  # "direct" | "agent"

    if mode == "agent":
        await run_agent_demo(issue_key)
    else:
        await run_agent_demo(issue_key)
        # await run_direct_demo(issue_key)


if __name__ == "__main__":
    asyncio.run(main())