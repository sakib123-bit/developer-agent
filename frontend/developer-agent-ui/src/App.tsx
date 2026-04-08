import { useState, useRef, useEffect } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

// ── Types ──────────────────────────────────────────────────────────────────────

type Chunk = {
  name: string;
  chunk_type: string;
  parent_class: string;
  file_path: string;
  start_line: number;
  end_line: number;
  score: number;
};

type Ticket = {
  key?: string;
  summary?: string;
  description?: string;
  status?: string;
  priority?: string;
  url?: string;
};

type AgentResult = {
  issue_key: string;
  ticket: Ticket;
  chunks: Chunk[];
  diff: string;
  explanation: string;
  error: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  result?: AgentResult;
  loading?: boolean;
};

// Step in the onboarding flow
type Step = "awaiting_repo" | "awaiting_ticket" | "chatting";

// ── Diff renderer ─────────────────────────────────────────────────────────────

type DiffRow =
  | { kind: "file"; content: string }
  | { kind: "hunk"; content: string }
  | { kind: "add"; content: string; newNum: number }
  | { kind: "remove"; content: string; oldNum: number }
  | { kind: "context"; content: string; oldNum: number; newNum: number };

function parseDiff(raw: string): DiffRow[] {
  const rows: DiffRow[] = [];
  let oldN = 0;
  let newN = 0;

  for (const line of raw.split("\n")) {
    if (line.startsWith("---") || line.startsWith("+++")) {
      rows.push({ kind: "file", content: line });
    } else if (line.startsWith("@@")) {
      const m = line.match(/@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
      if (m) { oldN = parseInt(m[1]); newN = parseInt(m[2]); }
      rows.push({ kind: "hunk", content: line });
    } else if (line.startsWith("-")) {
      rows.push({ kind: "remove", content: line.slice(1), oldNum: oldN++ });
    } else if (line.startsWith("+")) {
      rows.push({ kind: "add", content: line.slice(1), newNum: newN++ });
    } else {
      // context line (starts with space) or blank
      rows.push({ kind: "context", content: line.slice(1), oldNum: oldN++, newNum: newN++ });
    }
  }
  return rows;
}

// ── Split (before / after) view ───────────────────────────────────────────────

type SplitRow = {
  oldNum: number | null;
  newNum: number | null;
  oldContent: string | null; // null = empty cell
  newContent: string | null;
  kind: "context" | "change" | "hunk" | "file";
};

function buildSplitRows(rows: DiffRow[]): SplitRow[] {
  const out: SplitRow[] = [];
  let i = 0;
  while (i < rows.length) {
    const row = rows[i];
    if (row.kind === "file" || row.kind === "hunk") {
      out.push({ oldNum: null, newNum: null, oldContent: row.content, newContent: row.content, kind: row.kind });
      i++;
      continue;
    }
    if (row.kind === "context") {
      out.push({ oldNum: row.oldNum, newNum: row.newNum, oldContent: row.content, newContent: row.content, kind: "context" });
      i++;
      continue;
    }
    // Pair consecutive removes with adds
    const removes: Extract<DiffRow, { kind: "remove" }>[] = [];
    const adds:    Extract<DiffRow, { kind: "add" }>[]    = [];
    while (i < rows.length && rows[i].kind === "remove") {
      removes.push(rows[i] as Extract<DiffRow, { kind: "remove" }>);
      i++;
    }
    while (i < rows.length && rows[i].kind === "add") {
      adds.push(rows[i] as Extract<DiffRow, { kind: "add" }>);
      i++;
    }
    const maxLen = Math.max(removes.length, adds.length);
    for (let j = 0; j < maxLen; j++) {
      const rem = removes[j];
      const add = adds[j];
      out.push({
        oldNum: rem ? rem.oldNum : null,
        newNum: add ? add.newNum : null,
        oldContent: rem ? rem.content : null,
        newContent: add ? add.content : null,
        kind: "change",
      });
    }
  }
  return out;
}

function SplitView({ rows }: { rows: DiffRow[] }) {
  const splitRows = buildSplitRows(rows);
  return (
    <table className="diff-table split-table">
      <colgroup>
        <col style={{ width: 44 }} /><col style={{ width: 44 }} />
        <col /><col style={{ width: 44 }} /><col style={{ width: 44 }} /><col />
      </colgroup>
      <tbody>
        {splitRows.map((row, i) => {
          if (row.kind === "file") return (
            <tr key={i} className="diff-row-file">
              <td className="diff-gutter" colSpan={6} />
            </tr>
          );
          if (row.kind === "hunk") return (
            <tr key={i} className="diff-row-hunk">
              <td className="diff-gutter" colSpan={3} />
              <td className="diff-code diff-code-hunk" colSpan={3}>{row.newContent}</td>
            </tr>
          );
          const isCtx    = row.kind === "context";
          const hasOld   = row.oldContent !== null;
          const hasNew   = row.newContent !== null;
          return (
            <tr key={i} className={isCtx ? "diff-row-context" : "diff-row-change"}>
              <td className={`diff-gutter ${!isCtx && hasOld ? "diff-gutter-old" : ""}`}>{row.oldNum ?? ""}</td>
              <td className="diff-indicator split-indicator">{!isCtx && hasOld ? "−" : ""}</td>
              <td className={`diff-code split-cell ${!isCtx && hasOld ? "split-old" : ""}`}>{row.oldContent ?? ""}</td>
              <td className={`diff-gutter ${!isCtx && hasNew ? "diff-gutter-new" : ""}`}>{row.newNum ?? ""}</td>
              <td className="diff-indicator split-indicator">{!isCtx && hasNew ? "+" : ""}</td>
              <td className={`diff-code split-cell ${!isCtx && hasNew ? "split-new" : ""}`}>{row.newContent ?? ""}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function DiffView({ diff }: { diff: string }) {
  if (!diff.trim()) return null;
  const rows = parseDiff(diff);
  const [view, setView] = useState<"unified" | "split">("unified");

  return (
    <div className="diff-container">
      <div className="diff-header">
        <span>Generated Diff</span>
        <div className="diff-view-toggle">
          <button className={view === "unified" ? "active" : ""} onClick={() => setView("unified")}>Unified</button>
          <button className={view === "split"   ? "active" : ""} onClick={() => setView("split")}>Before / After</button>
        </div>
      </div>
      <div className="diff-table-wrap">
        {view === "unified" ? (
          <table className="diff-table">
            <tbody>
              {rows.map((row, i) => {
                if (row.kind === "file") return (
                  <tr key={i} className="diff-row-file">
                    <td className="diff-gutter" colSpan={3} />
                    <td className="diff-code diff-code-file">{row.content}</td>
                  </tr>
                );
                if (row.kind === "hunk") return (
                  <tr key={i} className="diff-row-hunk">
                    <td className="diff-gutter" colSpan={3} />
                    <td className="diff-code diff-code-hunk">{row.content}</td>
                  </tr>
                );
                if (row.kind === "remove") return (
                  <tr key={i} className="diff-row-remove">
                    <td className="diff-gutter diff-gutter-old">{row.oldNum}</td>
                    <td className="diff-gutter diff-gutter-new" />
                    <td className="diff-indicator">−</td>
                    <td className="diff-code">{row.content}</td>
                  </tr>
                );
                if (row.kind === "add") return (
                  <tr key={i} className="diff-row-add">
                    <td className="diff-gutter diff-gutter-old" />
                    <td className="diff-gutter diff-gutter-new">{row.newNum}</td>
                    <td className="diff-indicator">+</td>
                    <td className="diff-code">{row.content}</td>
                  </tr>
                );
                return (
                  <tr key={i} className="diff-row-context">
                    <td className="diff-gutter diff-gutter-old">{row.oldNum}</td>
                    <td className="diff-gutter diff-gutter-new">{row.newNum}</td>
                    <td className="diff-indicator" />
                    <td className="diff-code">{row.content}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : (
          <SplitView rows={rows} />
        )}
      </div>
    </div>
  );
}

// ── Result card ───────────────────────────────────────────────────────────────

type ApplyStatus = "idle" | "loading" | "done" | "error";

function ResultCard({ result, repoPath }: { result: AgentResult; repoPath: string }) {
  const { ticket, chunks, diff, explanation } = result;
  const [showChunks, setShowChunks]           = useState(false);
  const [applyStatus, setApplyStatus]         = useState<ApplyStatus>("idle");
  const [prUrl, setPrUrl]                     = useState("");
  const [applyError, setApplyError]           = useState("");

  const handleApply = async () => {
    setApplyStatus("loading");
    setPrUrl("");
    setApplyError("");
    try {
      const res = await fetch(`${API_BASE}/api/repo/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          issue_key:      result.issue_key,
          repo_path:      repoPath,
          diff:           diff,
          ticket_summary: ticket.summary ?? "",
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? res.statusText);
      if (!data.commit) throw new Error(data.error || "No changes were committed");
      setPrUrl(data.pr_url ?? "");
      setApplyStatus("done");
      if (data.error) setApplyError(data.error); // partial success (pushed but no PR token)
    } catch (err: unknown) {
      setApplyError(err instanceof Error ? err.message : String(err));
      setApplyStatus("error");
    }
  };

  return (
    <div className="result-card">
      {ticket?.summary && (
        <div className="ticket-info">
          <div className="ticket-meta">
            {ticket.key && (
              <a
                className="ticket-key"
                href={ticket.url ?? "#"}
                target="_blank"
                rel="noreferrer"
              >
                {ticket.key}
              </a>
            )}
            {ticket.status && (
              <span className="ticket-badge status">{ticket.status}</span>
            )}
            {ticket.priority && (
              <span className="ticket-badge priority">{ticket.priority}</span>
            )}
          </div>
          <div className="ticket-summary">{ticket.summary}</div>
        </div>
      )}

      {chunks.length > 0 && (
        <div className="chunks-section">
          <button
            className="toggle-btn"
            onClick={() => setShowChunks((v) => !v)}
          >
            {showChunks ? "▾" : "▸"} {chunks.length} relevant code chunk
            {chunks.length !== 1 ? "s" : ""}
          </button>
          {showChunks && (
            <ul className="chunks-list">
              {chunks.map((c, i) => {
                const label = c.parent_class
                  ? `${c.parent_class}.${c.name}`
                  : c.name;
                return (
                  <li key={i} className="chunk-item">
                    <span className="chunk-score">{c.score.toFixed(3)}</span>
                    <span className="chunk-label">{label}</span>
                    <span className="chunk-path">
                      {c.file_path}:{c.start_line}-{c.end_line}
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}

      <DiffView diff={diff} />

      {explanation && (
        <div className="explanation-section">
          <div className="explanation-header">Explanation</div>
          <p className="explanation-body">{explanation}</p>
        </div>
      )}

      {diff && (
        <div className="apply-section">
          {applyStatus === "idle" && (
            <button className="apply-btn" onClick={handleApply}>
              Apply Changes &amp; Create PR
            </button>
          )}
          {applyStatus === "loading" && (
            <span className="apply-status loading-text">
              Applying diff, committing, pushing...
            </span>
          )}
          {applyStatus === "done" && (
            <div className="apply-result">
              <span className="apply-success">Changes applied.</span>
              {prUrl && (
                <a className="pr-link" href={prUrl} target="_blank" rel="noreferrer">
                  View Pull Request
                </a>
              )}
              {applyError && (
                <span className="apply-warning">{applyError}</span>
              )}
            </div>
          )}
          {applyStatus === "error" && (
            <span className="apply-error">{applyError}</span>
          )}
        </div>
      )}
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────

const ISSUE_KEY_RE = /\b([A-Z][A-Z0-9]+-\d+)\b/;

function App() {
  const [step, setStep] = useState<Step>("awaiting_repo");
  const [repo, setRepo] = useState("");
  const [currentResult, setCurrentResult] = useState<AgentResult | null>(null);
  // Ref tracks which message index holds the live result card so we can update it in-place
  const resultMsgIdxRef = useRef<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hey! I'm Developer-Agent. Which repository should I work on? Paste a GitHub URL or a local path.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const addMessage = (msg: Message) =>
    setMessages((prev) => [...prev, msg]);

  const replaceLastMessage = (msg: Message) =>
    setMessages((prev) => [...prev.slice(0, -1), msg]);

  const updateMessageAt = (idx: number, msg: Message) =>
    setMessages((prev) => prev.map((m, i) => (i === idx ? msg : m)));

  // ── Step handlers ────────────────────────────────────────────────────────

  const handleRepoStep = async (text: string) => {
    addMessage({ role: "user", content: text });
    addMessage({
      role: "assistant",
      content: "Checking and indexing the repository... this may take a few minutes on the first run.",
      loading: true,
    });
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/repo/index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo: text }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? res.statusText);
      }

      const data: { local_path: string; status: string; files_indexed: number; commit: string } =
        await res.json();

      const statusMsg =
        data.status === "already_indexed"
          ? "Repo is already up to date — no re-indexing needed."
          : `Repo indexed successfully (${data.files_indexed} file${data.files_indexed !== 1 ? "s" : ""} processed).`;

      replaceLastMessage({
        role: "assistant",
        content: `${statusMsg}\n\nWhich Jira ticket should I work on?`,
      });

      setRepo(data.local_path);
      setStep("awaiting_ticket");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      replaceLastMessage({
        role: "assistant",
        content: `Failed to index repo: ${msg}`,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleTicketStep = async (text: string) => {
    const match = text.match(ISSUE_KEY_RE);

    // No ticket key found — just echo and let the user keep typing
    if (!match) {
      addMessage({ role: "user", content: text });
      addMessage({
        role: "assistant",
        content: "I didn't spot a ticket key in that. Drop one in (e.g. LL-1) and I'll get started.",
      });
      return;
    }

    const issueKey = match[1];
    addMessage({ role: "user", content: text });
    addMessage({
      role: "assistant",
      content: `On it! Fetching ${issueKey}, searching the codebase, and generating a diff...`,
      loading: true,
    });

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/agent/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ issue_key: issueKey, repo_path: repo }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? res.statusText);
      }

      const data: AgentResult = await res.json();

      // Replace the loading bubble with the result card, and record its index
      setMessages((prev) => {
        const idx = prev.length - 1;
        resultMsgIdxRef.current = idx;
        return [
          ...prev.slice(0, idx),
          {
            role: "assistant",
            content: `Here's what I found for ${issueKey}. Review the diff and apply when ready.`,
            result: data,
          },
        ];
      });

      setCurrentResult(data);
      setStep("chatting");

      addMessage({
        role: "assistant",
        content: "Ask me anything about the ticket, or say what you want changed and I'll update the diff right here.",
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      replaceLastMessage({
        role: "assistant",
        content: `Error: ${msg}`,
      });
    } finally {
      setLoading(false);
    }
  };

  // ── Chat step (after diff is generated) ──────────────────────────────────

  const handleChatStep = async (text: string, context: AgentResult) => {
    addMessage({ role: "user", content: text });
    addMessage({ role: "assistant", content: "Thinking...", loading: true });
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/agent/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticket:       context.ticket,
          chunks:       context.chunks,
          diff:         context.diff,
          explanation:  context.explanation,
          repo_path:    repo,
          user_message: text,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? res.statusText);
      }
      const data: { reply: string | null; diff: string | null; explanation: string | null } =
        await res.json();

      if (data.diff) {
        const updated: AgentResult = {
          ...context,
          diff:        data.diff,
          explanation: data.explanation ?? context.explanation,
        };
        setCurrentResult(updated);
        // Update the original result card in-place (feedback loop)
        if (resultMsgIdxRef.current !== null) {
          updateMessageAt(resultMsgIdxRef.current, {
            role: "assistant",
            content: `Here's what I found for ${context.issue_key}. Review the diff and apply when ready.`,
            result: updated,
          });
        }
        replaceLastMessage({
          role: "assistant",
          content: "Diff updated above. Keep refining or apply when ready.",
        });
      } else {
        replaceLastMessage({
          role: "assistant",
          content: data.reply ?? "Done.",
        });
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      replaceLastMessage({ role: "assistant", content: `Error: ${msg}` });
    } finally {
      setLoading(false);
    }
  };

  // ── Send ─────────────────────────────────────────────────────────────────

  const isRepoLike = (t: string) =>
    t.startsWith("/") || t.startsWith("~") || t.includes("github.com") || t.includes("://");

  const sendMessage = () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");

    if (step === "awaiting_repo") {
      handleRepoStep(text);
      return;
    }

    // Switch repo at any time when a URL/path is entered
    if (!text.match(ISSUE_KEY_RE) && isRepoLike(text)) {
      handleRepoStep(text);
      return;
    }

    if (step === "awaiting_ticket") {
      handleTicketStep(text);
    } else if (step === "chatting") {
      // New ticket key → run agent again
      if (text.match(ISSUE_KEY_RE)) {
        handleTicketStep(text);
      } else if (currentResult) {
        handleChatStep(text, currentResult);
      }
    }
  };

  // ── Placeholder text ─────────────────────────────────────────────────────

  const placeholder = loading
    ? "Waiting for agent..."
    : step === "awaiting_repo"
    ? "Enter repo path or GitHub URL..."
    : step === "awaiting_ticket"
    ? "Enter a Jira ticket key, e.g. LL-1"
    : "Ask about the ticket or request code changes...";

  return (
    <div className="app">
      <header className="header">
        <h2>Developer-Agent</h2>
        {repo && (
          <div className="repo-context">
            <span className="repo-label">Repo:</span>
            <span className="repo-value">{repo}</span>
            <button
              className="repo-change-btn"
              onClick={() => {
                if (loading) return;
                setStep("awaiting_repo");
                addMessage({
                  role: "assistant",
                  content: "Sure, which repo should I switch to?",
                });
              }}
            >
              change
            </button>
          </div>
        )}
      </header>

      <div className="chat-container">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="bubble">
              <span className={msg.loading ? "loading-text" : ""}>
                {msg.content}
              </span>
              {msg.result && <ResultCard result={msg.result} repoPath={repo} />}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="input-container">
        <input
          type="text"
          placeholder={placeholder}
          value={input}
          disabled={loading}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button onClick={sendMessage} disabled={loading}>
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}

export default App;
