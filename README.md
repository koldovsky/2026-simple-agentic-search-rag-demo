# Simple Agentic Search RAG Demo (2026)

A small Node.js console app that answers questions about a document using **agentic search RAG**:
an LLM agent searches the document as a tool, checks whether the results actually answer the question,
and **searches again with a reformulated query** if they don't — up to a fixed budget, after which it
answers with what it found or honestly gives up.

The sample document ([doc/context.txt](doc/context.txt)) is a Ukrainian FAQ about IT STEP University (Lviv).

```
$ npm run ask -- "Чи можна навчатися дистанційно і скільки це коштує?"

🔎 search 1/3: "дистанційне навчання вартість навчання дистанційна форма" → chunks 5, 6, 1, 23
🔎 search 2/3: "онлайн дистанційно формат занять віддалене навчання" → chunks 26, 31, 34, 25
💬 answer (2 searches, 3 LLM calls, 3.3k tokens)

Основні програми університету проводяться офлайн, хоча для студентів є онлайн-курси та тренінги... [chunk 26]
```

## How it works

```
                 ┌──────────────── agent loop (max 6 LLM calls) ────────────────┐
question ──► LLM ──► search_document(query) ──► hybrid search ──► new chunks ──► LLM ──► answer
                 ▲                                                                │
                 └──── results not relevant? reformulate & search again (max 3) ──┘
```

1. **Indexing** — the document is split into ~500-char chunks, each chunk is embedded with
   OpenAI `text-embedding-3-small`, and everything is stored locally in `data/index.json`.
2. **Hybrid search** — every query runs BM25 keyword search (with a Ukrainian stemmer and stop words)
   and vector similarity search, and the two rankings are combined.
3. **Agent** — an LLM (`gpt-6.1-sol`, low reasoning effort) calls the `search_document` tool, reads the
   results and decides whether they contain the answer. If not, it searches again with different wording.
   - **Search budget: 3 searches** (1 initial + 2 re-searches), enforced in code, not by the model.
     Each search only returns chunks the agent hasn't seen yet, so 3 searches cover ~12 distinct chunks
     (~30% of the document) with 3 different phrasings. If that misses, the answer is almost certainly
     not in the document; more attempts mostly add cost and the risk of stretching half-relevant text.
   - When the budget is used up, the tool is removed and the model must answer or say it doesn't know.
   - The system prompt is in [src/system-prompt.js](src/system-prompt.js).

## Tech stack

| Concern | Library |
|---|---|
| Agent loop, LLM and embeddings | [Vercel AI SDK](https://ai-sdk.dev) (`ai`, `@ai-sdk/openai`) — `ToolLoopAgent` |
| Keyword + vector + hybrid search | [Orama](https://orama.com) (`@orama/orama`, `@orama/stemmers`) |
| Ukrainian stop words | [`stopword`](https://github.com/fergiemcdowall/stopword) |
| Chunking | `@langchain/textsplitters` (`RecursiveCharacterTextSplitter`) |
| CLI / config | `commander`, `dotenv`, `zod` |

## Getting started

Requires **Node.js 20+** and an **OpenAI API key**.

```bash
git clone https://github.com/koldovsky/2026-simple-agentic-search-rag-demo.git
cd 2026-simple-agentic-search-rag-demo
npm install
cp .env.example .env   # then put your key into .env
npm run index          # chunk + embed the document → data/index.json
npm start              # interactive agent chat
```

The index is also built automatically on first use and rebuilt whenever `doc/context.txt` or the
embedding model changes.

## Commands

| Command | What it does |
|---|---|
| `npm start` | Interactive chat with the agent (empty line or `exit` to quit) |
| `npm run ask -- "question"` | Ask the agent one question, with a trace of every search |
| `npm run index` | Rebuild the search index |
| `npm run search -- "query"` | Raw hybrid search **without** the agent, showing vector and keyword ranks |
| `npm run search:vector -- "query"` | Raw vector-only search |
| `npm run search:keyword -- "query"` | Raw BM25 keyword search (no API calls, works offline once indexed) |

Good questions to try:

- `"Скільки коштує навчання на комп'ютерних науках?"` — answered after one search
- `"Where can I live while studying?"` — English question; the agent searches in Ukrainian and answers in English
- `"Чи можна навчатися дистанційно і скільки це коштує?"` — needs a re-search
- `"Яка зарплата ректора?"` — not in the document: uses all 3 searches, then gives up honestly

## Configuration

Set in `.env` (see [.env.example](.env.example)):

| Variable | Default | |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `LLM_MODEL` | `gpt-6.1-sol` | e.g. `gpt-6-luna` for a cheaper, faster agent |
| `LLM_REASONING_EFFORT` | `low` | `none` (Luna only), `low`, `medium`, `high`, … |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Changing it triggers a re-index |

Search and agent tuning constants (chunk size, `MAX_SEARCHES`, `MAX_STEPS`, hybrid weights, …) live in
[src/config.js](src/config.js). A typical question costs about $0.01–0.05 with `gpt-6.1-sol`.

The app never prints the API key, and error messages are redacted, so it is safe to run while screen sharing.

## Project structure

```
doc/context.txt        source document
src/index.js           CLI: chat, ask, index, search
src/agent.js           agent: search tool, search budget, ToolLoopAgent
src/system-prompt.js   system prompt
src/search-engine.js   Orama index: create, save/load, search modes
src/indexer.js         chunking + embedding + index freshness check
src/embeddings.js      OpenAI embeddings via the AI SDK
src/config.js          settings, .env loading, secret redaction
```

## Step by step: how this demo was built

The git history doubles as a tutorial — check out any commit to see that stage:

| Commit | Stage |
|---|---|
| `138bd6e` | Plain embedding search: hand-written chunker, cosine similarity, JSON store |
| `a5ec120` | Hybrid search: hand-written BM25 + Reciprocal Rank Fusion |
| `8102b5e` | Same features on well-known libraries (Orama, LangChain splitter, dotenv, commander) |
| `7ba3963` | Agentic RAG with a **hand-written** agent loop on the OpenAI Responses API |
| `2eaac3c` | Agent loop refactored onto the Vercel AI SDK `ToolLoopAgent` |

```bash
git checkout 7ba3963   # e.g. see the hand-written agent loop
git checkout master    # back to the latest version
```
