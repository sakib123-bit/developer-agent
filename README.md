# Developer Agent

An AI-powered developer agent that takes a Jira ticket, finds the relevant code in your repository, generates a unified diff, and opens a GitHub PR — all from a chat UI.

## How it works

1. **Paste a repo URL or local path** — the agent clones it and indexes the code into Pinecone using Google Gemini embeddings.
2. **Enter a Jira ticket key** (e.g. `LL-1`) — the agent fetches the ticket, searches indexed code semantically, reads the relevant files, and generates a minimal unified diff with GPT-4o.
3. **Review the diff** in the built-in before/after viewer, then click **Apply Changes & Create PR** to commit and open a GitHub PR.
4. **Iterate** — ask follow-up questions about the ticket or request code changes in plain English; the diff updates in-place.

## Stack

| Layer | Tech |
|---|---|
| Agent | [LangGraph](https://github.com/langchain-ai/langgraph) + GPT-4o |
| Code search | [Pinecone](https://pinecone.io) + Google Gemini embeddings |
| Jira | [FastMCP](https://github.com/jlowin/fastmcp) stdio server |
| API | FastAPI + uvicorn |
| Frontend | React 19 + TypeScript + Vite |

## Project structure

```
agent/          # LangGraph agent (fetch ticket → search → read → diff)
api/            # FastAPI server + git/PR operations
indexer/        # Code parser, embedder, Pinecone pipeline
frontend/       # React chat UI
jira_mcp_server.py  # Jira MCP server (FastMCP)
client.py       # MCP client wrapper
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
4. Click **Apply Changes & Create PR** — the agent commits the changes on a `fix/<ticket>` branch and opens a PR.
5. Continue the conversation to refine the diff or ask about the ticket.
