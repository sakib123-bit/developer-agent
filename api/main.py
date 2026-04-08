import asyncio
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path, override=True)

from agent.sdlc_agent import run_agent
from indexer.pipeline import index_repo
from api.git_ops import apply_diff_and_create_pr

app = FastAPI(title="Developer Agent API")

_REQUIRED_KEYS = ["OPENAI_API_KEY", "JIRA_URL", "JIRA_EMAIL", "JIRA_API_TOKEN",
                  "GOOGLE_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]

@app.on_event("startup")
async def check_env():
    missing = [k for k in _REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        print(f"\n[startup] WARNING: missing env vars: {', '.join(missing)}")
        print(f"[startup] .env loaded from: {_env_path} (exists={_env_path.exists()})\n")
    else:
        print(f"\n[startup] All required env vars present. .env: {_env_path}\n")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexRequest(BaseModel):
    repo: str  # URL or local path


class GenerateRequest(BaseModel):
    issue_key: str
    repo_path: str = ""


class ApplyRequest(BaseModel):
    issue_key: str
    repo_path: str
    diff: str
    ticket_summary: str = ""


class ChatRequest(BaseModel):
    ticket: dict
    chunks: list
    diff: str
    explanation: str
    repo_path: str = ""
    user_message: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/repo/index")
async def index_repo_endpoint(request: IndexRequest):
    if not request.repo.strip():
        raise HTTPException(status_code=400, detail="repo is required")
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, index_repo, request.repo)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/repo/apply")
async def apply_changes(request: ApplyRequest):
    if not request.diff.strip():
        raise HTTPException(status_code=400, detail="diff is required")
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            apply_diff_and_create_pr,
            request.repo_path,
            request.diff,
            request.issue_key,
            request.ticket_summary,
        )
        # Always return 200 with the result dict; frontend reads result.error
        return result
    except Exception as e:
        return {"error": str(e), "branch": None, "commit": None, "pr_url": None}


@app.post("/api/agent/chat")
async def agent_chat(request: ChatRequest):
    import re
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    if not request.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )

    ticket = request.ticket
    chunk_summary = "\n".join(
        f"[{c.get('score', 0):.3f}] {c.get('file_path', '')} → "
        f"{c.get('parent_class', '') + '.' if c.get('parent_class') else ''}{c.get('name', '')} "
        f"(lines {c.get('start_line', '')}-{c.get('end_line', '')})"
        for c in request.chunks
    )

    system = SystemMessage(content=(
        "You are an expert developer assistant. "
        "You have context about a Jira ticket and a generated code diff. "
        "If the user asks a question about the ticket (priority, status, description, etc.) answer it directly and conversationally. "
        "If the user asks to modify, update, or improve the generated code/diff, produce a new complete unified diff "
        "followed by 'EXPLANATION:' and your explanation. "
        "When generating a diff use the exact unified diff format: --- a/path, +++ b/path, @@ hunks. "
        "Never mix a conversational reply with a diff — either produce a plain text answer OR a diff+explanation."
    ))

    human = HumanMessage(content=f"""JIRA TICKET
-----------
Key        : {ticket.get('key', '')}
Summary    : {ticket.get('summary', '')}
Description: {ticket.get('description', '')}
Priority   : {ticket.get('priority', '')}
Status     : {ticket.get('status', '')}

RELEVANT CODE CHUNKS
--------------------
{chunk_summary}

CURRENT DIFF
------------
{request.diff or "(no diff yet)"}

CURRENT EXPLANATION
-------------------
{request.explanation or "(none)"}

USER REQUEST
------------
{request.user_message}
""")

    response = await llm.ainvoke([system, human])
    raw = response.content.strip()

    # Detect if the model produced a new diff
    if ("--- a/" in raw or "+++ b/" in raw) and "@@ " in raw:
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\n?```\s*$", "", raw)
        raw = raw.strip()
        if "EXPLANATION:" in raw:
            parts = raw.split("EXPLANATION:", 1)
            new_diff = parts[0].strip()
            new_explanation = parts[1].strip()
        else:
            new_diff = raw
            new_explanation = ""
        return {"reply": None, "diff": new_diff, "explanation": new_explanation}

    return {"reply": raw, "diff": None, "explanation": None}


@app.post("/api/agent/generate")
async def generate(request: GenerateRequest):
    if not request.issue_key.strip():
        raise HTTPException(status_code=400, detail="issue_key is required")

    result = await run_agent(request.issue_key, request.repo_path)

    if result.get("error") and not result.get("explanation"):
        raise HTTPException(status_code=500, detail=result["error"])

    return {
        "issue_key":    result.get("issue_key", ""),
        "ticket":       result.get("ticket", {}),
        "chunks":       result.get("chunks", []),
        "diff":         result.get("diff", ""),
        "explanation":  result.get("explanation", ""),
        "error":        result.get("error", ""),
    }
