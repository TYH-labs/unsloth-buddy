# Hermes Agent: Memory and Scalability Design 

The `hermes-agent` employs a highly optimized, multi-tiered architecture to allow agents to "grow" and handle increasingly complex, long-running tasks. Its design is particularly well-suited for high-throughput, autonomous operation, heavily prioritizing **LLM prefix caching** and **token efficiency**.

Here is a walkthrough of how `hermes-agent` preserves information, the rationales behind its design choices, and areas where future agent frameworks can learn and improve.

## 1. The Three Tiers of Memory

### Tier 1: Persistent Curated Facts ([tools/memory_tool.py](file:///Users/hliu/github/hermes-agent/tools/memory_tool.py))
Hermes separates *ephemeral task state* from *permanent facts*. The agent is provided a `memory` tool that lets it perform CRUD operations (`add`, `replace`, `remove`) on two explicit stores:
* **`USER.md`**: Tracks user communication styles, preferences, role, and timezone.
* **`MEMORY.md`**: Tracks environment facts, constraints, API quirks, and project conventions.

**Implementation Highlight (The Frozen Snapshot):**
When a session begins, `hermes-agent` reads these files and injects them into the system prompt. If the agent updates its memory mid-session, **it immediately persists to disk, but the system prompt is NOT updated.** 
* **Rationale:** By keeping the system prompt permanently static across a given session, it guarantees an unbroken **Prefix Cache** (e.g., Anthropic Prompt Caching). If memory updates continually changed the system prompt, every turn would experience a cache miss, exponentially increasing costs on large codebases.

### Tier 2: Real-time Context Compression ([agent/context_compressor.py](file:///Users/hliu/github/hermes-agent/agent/context_compressor.py))
As the agent performs consecutive tool calls, the context window fills up rapidly. The `ContextCompressor` triggers automatically when the context exceeds a predefined percentage (e.g., 50%) of the model's limit.

**Implementation Highlight (The "Handoff" Summary):**
1. **Pre-pass Pruning:** Before calling any LLM, it cheaply truncates old tool outputs (e.g., replacing large CLI responses with `[Old tool output cleared]`), saving thousands of tokens instantly.
2. **Tail Budget Protection:** It protects the most recent ~20K tokens (the "Tail") and the system prompt (the "Head") so the agent doesn't lose immediate context. 
3. **Structured Summary via Auxiliary Client:** The middle turns are sent to a less expensive, faster LLM. The prompt explicitly instructs the LLM to act as a *Summarization Agent* transferring state to a "different assistant."
4. **Iterative Updates:** Rather than summarizing from scratch every time, it pipes the *previous* summary into the next compression run to iteratively mutate the state.

### Tier 3: Immutable Event Store ([hermes_state.py](file:///Users/hliu/github/hermes-agent/hermes_state.py) / [gateway/session.py](file:///Users/hliu/github/hermes-agent/gateway/session.py))
Beneath the agent's immediate awareness, all transcripts and tool calls are logged to a local SQLite database using Write-Ahead Logging (WAL). 
* An FTS5 (Full-Text Search) virtual table indexes every message natively in sqlite.
* The agent possesses a `session_search_tool` that it can use to query this database. If it realizes it needs an output from 3 days ago, it can essentially "Google" its own history.

---

## 2. Handling Complexity: How the Agent "Grows" ([delegate_tool.py](file:///Users/hliu/github/hermes-agent/tools/delegate_tool.py) & [todo_tool.py](file:///Users/hliu/github/hermes-agent/tools/todo_tool.py))

A pure context window compression handles long history, but to handle parallel tasks or massive codebases, Hermes employs two additional techniques:

### Episodic Task Breakdown ([todo_tool.py](file:///Users/hliu/github/hermes-agent/tools/todo_tool.py))
When an agent faces a non-trivial task, it does *not* write its plan directly into the chat (which consumes tokens endlessly as the chat grows). Instead, it uses a CRUD `todo` tool.
* This is an **in-memory task list** attached directly to the AIAgent instance. 
* The brilliance of this tool is that its state **survives context compression**. During a compression event, all "pending" and "in_progress" to-dos are dynamically re-injected as a special summary message, completely preventing the "forgetting what I was doing" failure mode commonly seen in LLMs after a long search task.

### Subagent Delegation ([delegate_tool.py](file:///Users/hliu/github/hermes-agent/tools/delegate_tool.py))
This is the ultimate scaling mechanism. To prevent the parent context window from drowning in granular CLI logs, the parent agent can spawn parallel `AIAgent` children.
* **Hermetically Sealed:** Children receive an entirely fresh context, a focused goal, and a subset of tools. The parent's memory context is shielded.
* **Restricted Tools:** Children are intentionally denied access to `memory`, `clarify` (so they cannot bother the human user), and `delegate_task` (preventing infinite subagent recursion).
* **Summary Returns:** When the child finishes, only its specialized summary (e.g., "I successfully fixed `app.tsx` and ran the tests. Here is what I did...") is returned to the parent, effectively solving token explosions for massive search/replace jobs.

---

## 3. Rationales: Why they built it this way

* **The Anti-Bloat Strategy**: Older agent frameworks just aggressively shoved everything into the vector database or context window. Hermes caps its curated memory to strict *character limits* (~2200 chars) instead of token limits to remain model-agnostic. 
* **Security & Prompt Injection Prevention:** [memory_tool.py](file:///Users/hliu/github/hermes-agent/tools/memory_tool.py) scans memory mutations using a regex blocklist (`_MEMORY_THREAT_PATTERNS`) to prevent payload exfiltration (e.g., `ignore previous instructions` or `curl $AWS_ACCESS_KEY`). Because memory is injected into the system prompt, an attacker tricking the agent into saving a malicious payload would otherwise poison all future sessions.
* **Tool-Call Integrity:** When compressing context, one of the easiest ways to break an OpenAI/Anthropic API call is to accidentally separate a `tool_call` from its matching `tool_result`. Hermes walks the array and forcefully injects "stub results" for orphaned calls so the API never hard-crashes.

---

## 4. How Other Frameworks Can Learn From This

1. **Adopt "Frozen Snapshot" System Prompts:** If you are building an agent, do not dynamically change the system instructions mid-conversation to reflect new "memories." Save it to disk or a DB, let the tool return `"Memory saved"`, and load it on the *next* conversation. Your API costs will drop by 80-90% due to perfect prompt caching.
2. **Isolate Tool Groups During Truncation:** [context_compressor.py](file:///Users/hliu/github/hermes-agent/agent/context_compressor.py)::_align_boundary_backward() demonstrates an incredibly robust way to handle context limits without raising JSON schema errors from the LLM provider.
3. **Structured Context Summaries over Conversational Summaries:** Hermes enforces a strict markdown schema for compression:
    * `## Goal`
    * `## Progress (Done / In Progress / Blocked)`
    * `## Resolved Questions` 
    * `## Pending User Asks`
    * `## Remaining Work` *(Framed as context, not instructions, to prevent the LLM from getting trapped in a loop)*.

---

## 5. How it can do better

While Hermes has a highly durable state machine, it has limitations that could be improved in the next generation of agents:

1. **Lack of Autonomous Offline Reflection**: 
   Currently, the agent must *actively decide* to call the `memory` tool to remember something. If the model forgets or ignores the instruction, knowledge is lost. 
   **Improvement**: Implement an offline cron-job that runs over the `SessionDB` after a conversation goes idle. Use an auxiliary LLM to extract facts, skills, and preferences and merge them into the memory files without consuming the main agent's time.

2. **No Semantic (Vector) Search on Core Memory**: 
   Because `MEMORY.md` and `USER.md` are blindly injected into the system prompt, they must remain short (capped by character limits). 
   **Improvement**: Move to an architecture like **MemGPT** where memory is partitioned into "Core" (injected every turn) and "Archival" (vector-searched).

3. **Skill vs. Fact Blurring**: 
   Hermes has both a `memory_tool` (for facts) and a `skill_tool` (to write executable python scripts / shell routines). Agents often confuse the two, writing procedural coding steps into the memory file where it wastes tokens, rather than saving them as reusable files.
   **Improvement**: Introduce a memory "router" that intercepts a generic `save_knowledge` tool call and automatically classifies it into a semantic fact, an actionable script, or a user preference.
