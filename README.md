# Developer Agent

An AI-powered developer agent that takes a Jira ticket, finds the relevant code in your repository, generates a unified diff, and opens a GitHub PR — all from a chat UI.

## MCP-Based Architecture

The Developer Agent is built on the [Model Context Protocol (MCP)](https://modelcontextprotocol.io). It has been re-architected to use standardized MCP tools for all its operations, moving away from hard-coded integrations.

- **Unified Tool Interface**: All capabilities—fetching Jira tickets, searching code, reading files, and creating PRs—are now implemented as MCP tools.
- **Decoupled Tools**: Tools are implemented as standalone MCP servers (e.g., `jira_mcp_server.py`), allowing them to be developed and tested independently.
- **Dynamic Discovery**: The agent uses an MCP client (`client.py`) to connect to these servers at runtime, discover available tools, and provide them to the LLM.
- **Generic Orchestration**: The core agent logic is now a generic orchestrator that can work with any MCP-compliant server without modification.

## How it works

1. **Tool Discovery**: On startup, the agent connects to the configured MCP servers and discovers their available tools.
2. **Context Gathering**: When a Jira ticket key is provided, the agent uses MCP tools to fetch the ticket description and comments.
3. **Code Search & Analysis**: The agent uses semantic search tools to find relevant code in Pinecone and file-reading tools to understand the context.
4. **Diff Generation & PR**: After generating a diff with GPT-4o, the agent uses Git-related MCP tools to apply the changes, commit them, and open a GitHub PR.
5. **Iterate**: Users can request changes in plain English, and the agent will re-invoke the necessary MCP tools to update the diff.

## Stack

| Layer | Tech |
|---|---|
| Agent | [LangGraph](https://github.com/langchain-ai/langgraph) + GPT-4o |
| MCP Client | Python MCP SDK |
| MCP Servers | [FastMCP](https://github.com/jlowin/fastmcp) |
| Code search | [Pinecone](https://pinecone.io) + Google Gemini embeddings |
| API | FastAPI + uvicorn |
| Frontend | React 19 + TypeScript + Vite |

## Project structure

```
agent/          # LangGraph agent (orchestrates MCP tool calls)
api/            # FastAPI server
indexer/        # Code parser, embedder, Pinecone pipeline
frontend/       # React chat UI
jira_mcp_server.py  # Jira MCP server (FastMCP)
client.py       # MCP client wrapper
mcp_agent.py    # Generic MCP agent implementation
```

## Setup

### 1. Install dependencies

```bash
uv sync
cd frontend/developer-agent-ui && npm install
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Required variables:

```
OPENAI_API_KEY=
GOOGLE_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
JIRA_URL=https://your-org.atlassian.net
JIRA_EMAIL=you@example.com
JIRA_API_TOKEN=
GITHUB_TOKEN=          # needs repo scope for PR creation
```

### 3. Run

**Backend:**
```bash
uv run uvicorn api.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend/developer-agent-ui && npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Usage

1. Paste a GitHub repo URL (e.g. `https://github.com/org/repo`) or a local path.
2. Enter a Jira ticket key when prompted.
3. Review the generated diff in the Unified or Before/After view.
4. Click **Apply Changes & Create PR** — the agent uses MCP tools to commit the changes and open a PR.
5. Continue the conversation to refine the diff or ask about the ticket.

## Generic MCP Agent

In addition to the specialized developer workflow, you can run a generic MCP agent that provides a direct interface to any MCP server:

1. **Configure the server**: Ensure the MCP server you want to use (e.g., `jira_mcp_server.py`) is available and configured in your `.env`.
2. **Run the agent**:
   ```bash
   uv run mcp_agent.py --server-cmd "uv run jira_mcp_server.py"
   ```
3. **Interact**: The agent will connect to the server, list available tools, and wait for your instructions in the terminal.