# Introduction to LangChain — Course Notes

A hands-on course for building AI agents with **LangChain** and **LangGraph** — from a single chat model call to production-ready, multi-agent systems with middleware, human-in-the-loop approval, and MCP integration.

---

## 📚 Table of Contents

- [Chapter 1 — The Basics of Agent Building](#chapter-1--the-basics-of-agent-building)
- [Chapter 2 — Advanced Agent Techniques](#chapter-2--advanced-agent-techniques)
- [Chapter 3 — Middleware & Deployment-Ready Agents](#chapter-3--middleware--deployment-ready-agents)
- [Setup Requirements](#setup-requirements)

---

## Chapter 1 — The Basics of Agent Building

### 1.1 Models and Prompts
- The **model** is the "thinking brain" of an agent — initialize it with LangChain's `init_chat_model` function.
- Interacting with a model = invoke it → get back an **AIMessage** (contains `.content` + metadata like token usage).
- Customize model behavior with parameters:
  - `temperature` — controls randomness/creativity
  - `max_tokens` — limits response length
  - `timeout` — max wait time for a response
  - `max_retries` — retry attempts on failure
- LangChain is **model-agnostic** — swapping OpenAI, Claude, or Gemini is just a config change. Some newer models may require the provider's own LangChain library instead of `init_chat_model`.
- API keys are stored in a `.env` file.
- **`create_agent`** (built on LangGraph) is the core abstraction used throughout the course.
- Agents are invoked with a dictionary of messages (`HumanMessage`, etc.) and return the full conversation as a list of messages.
- **Streaming** tokens reduces perceived latency for slow agent responses.
- **System prompts** are the easiest way to customize agent behavior — including few-shot examples or output structure.
- **Structured output schemas** let you extract specific fields instead of parsing free text.

### 1.2 Tools
- Tools = what separates an **agent** from a simple chatbot. They let the agent take actions and react to results (the **ReAct** pattern).
- Define a tool with the `@tool` decorator; name/description default to the function name/docstring but can be overridden.
- Agent tool-calling flow: `HumanMessage` → `AIMessage` (with `tool_calls`) → `ToolMessage` (tool's result) → final `AIMessage`.
- Example: adding a **Tavily web search** tool lets the agent answer questions beyond its training cutoff (e.g., current events).
- **LangSmith** tracing gives a clean, end-to-end view of agent runs (latency, token usage, tool calls). Free tier: 5,000 traces/month. Optional but very useful for debugging.

### 1.3 Short-Term Memory
- By default, agents have **no memory** between invocations — state resets each run.
- Add memory using a **checkpointer** (e.g., `InMemorySaver` from LangGraph).
- A `thread_id` groups messages into the same conversation so the agent can recall prior context.
- Custom fields (e.g., `user_id`) can also be tracked in agent state.

### 1.4 Multimodal Messages
- Agents can accept **image** and **audio** input, not just text.
- Files are Base64-encoded and passed as a message with `type: image` (or audio) alongside a text message.
- ⚠️ Multimodal inputs can be used for **prompt injection** (e.g., hidden instructions in audio) — be aware of this risk.

### 1.5 Project: Personal Chef Agent
- Combines everything from Chapter 1: web search tool + system prompt + memory (checkpointer).
- Bonus: accept an image of your fridge/pantry to suggest recipes.
- Introduces **LangGraph Studio** for a local debugging UI:
  1. Convert your notebook agent into a `.py` file.
  2. Add a `langgraph.json` file pointing to your agent and `.env`.
  3. Run `langgraph dev` (or `uv run langgraph dev` if using `uv`).

---

## Chapter 2 — Advanced Agent Techniques

### 2.1 MCP (Model Context Protocol)
- MCP is an **open protocol by Anthropic** that standardizes how LLM apps connect to tools/data — think "USB for AI tools."
- Architecture: **MCP Host** (your agent) → **MCP Client** → **MCP Server** (exposes tools, resources, prompts).
- Build your own MCP server, or connect to public ones (e.g., a "time" server, Kiwi travel search).
- In LangChain: use `langchain_mcp_adapters.client.MultiServerMCPClient`, specifying a `transport` (`stdio` or `streamable_http`).
- Benefit: MCP servers are reusable across projects and compatible with other AI apps (chatbots, IDEs), not just LangChain.

### 2.2 Custom State and Context
- **Context** = static, read-only info passed at runtime (e.g., user role, preferred language) via a context schema (dataclass).
- **State** = mutable info the agent can read *and update* during a run (e.g., learned preferences).
- Both are accessed inside tools via **tool runtime** — not passed directly to the model (to avoid context-window overload).
- Custom state fields extend the default message-list state.

### 2.3 Multi-Agent Systems
- For complex, long workflows, split responsibilities across **specialized sub-agents** instead of overloading one agent.
- Common pattern: **Supervisor architecture** — a main orchestrating agent calls sub-agents as tools.
- Tracing (LangSmith) is essential here since sub-agent internals aren't visible in simple print statements.

### 2.4 Project: Destination Wedding Planner
- Multi-agent system with:
  - **Travel agent** (Kiwi MCP server for flights)
  - **Venue agent** (web search)
  - **DJ/music agent** (SQL database query tool)
  - **Coordinator agent** (manages state, delegates tasks, compiles results)

---

## Chapter 3 — Middleware & Deployment-Ready Agents

### 3.1 What Is Middleware?
- Middleware = functions inserted into the model/tool loop to customize agent execution (e.g., content filters, human approval gates).
- Key to turning a hobby project into a production-ready agent.

### 3.2 Managing Long Conversations
- Long message histories overflow the context window → slower, costlier, worse performance.
- Two middleware-based solutions:
  1. **Summarization middleware** (built-in) — summarizes older messages, keeps the last N.
  2. **Custom trimming middleware** — using `before_agent` / `after_agent` / `before_model` / `after_model` decorators (**node-style middleware**) to remove/filter messages (e.g., stripping tool messages).

### 3.3 Human-in-the-Loop (HITL)
- Use cases: approving sensitive actions, adding missing content, debugging.
- Built-in HITL middleware lets you flag specific tools as requiring approval (`True`/`False` per tool; default is auto-approved).
- When a flagged tool is called, the agent **interrupts** and returns an `interrupt` object.
- Resume the agent with a `Command`:
  - **Approve** — run the tool call as-is.
  - **Reject** — cancel, optionally with feedback.
  - **Edit** — modify the tool call arguments, then auto-approve.
- Must reuse the same `thread_id` to resume correctly.

### 3.4 Dynamic Agents
- **Wrap-style middleware** (via `wrap_model_call` decorator) lets you modify the model itself at runtime — system prompt, tools, or even the underlying model.
- Examples covered:
  - **Dynamic prompts** — respond in the user's preferred language (via `@dynamic_prompt`).
  - **Dynamic tools** — give internal users more tools (e.g., database access) than external users.
  - **Dynamic model switching** — use a smaller/cheaper model by default, switch to a larger model for long conversations.
- Decision matrix:
  | Middleware type | Access via | Use case |
  |---|---|---|
  | Wrap-style | `ModelRequest` | Change prompt/tools/model dynamically |
  | Node-style | `state` + `runtime` | Modify state (e.g., trim messages) |
  | Tool calls | `tool_runtime` | Agent reads/writes state or context |

### 3.5 Project: Email Assistant
- Combines dynamic middleware (auth-gated tools/prompts) + human-in-the-loop (approval before sending emails).
- Flow: authenticate → read inbox → draft reply → require human approval → send.

### 3.6 Bonus: Agent Chat UI
- A free, open-source, plug-and-play chat interface for demoing agents (**agentchat.vercel.app**).
- Connects to a locally running **LangGraph Studio** instance (default port `2024`).
- Supports hiding/showing tool calls, image/PDF uploads, and a clean HITL approve/edit/reject flow.
- Fully open source and customizable (name, logo, branding) — clone the repo and run locally.

---

## Setup Requirements

- Python environment (recommend [`uv`](https://github.com/astral-sh/uv) for dependency management)
- A `.env` file containing your API keys, e.g.:
  ```
  OPENAI_API_KEY=sk-...
  GROQ_API_KEY=gsk-...
  GOOGLE_API_KEY=...
  TAVILY_API_KEY=tvly-...
  LANGSMITH_API_KEY=lsv2-...
  ```
- `langgraph.json` file for running LangGraph Studio locally (`langgraph dev` or `uv run langgraph dev`)
- Optional: a free [LangSmith](https://smith.langchain.com/) account for tracing (5,000 traces/month on free tier)

> ⚠️ **Never commit your `.env` file.** Make sure it's listed in `.gitignore` before pushing to GitHub.

---

## Course Structure Summary

| Chapter | Focus |
|---|---|
| 1 | Models, prompts, tools, memory, multimodal input |
| 2 | MCP, custom state/context, multi-agent systems |
| 3 | Middleware, human-in-the-loop, dynamic agents, deployment |