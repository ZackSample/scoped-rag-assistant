# IMM Troubleshooting Assistant

A chat assistant for injection molding machine (IMM) technicians. Pick a
machine, ask a question, and get an answer grounded in that machine's own
manuals, with every claim cited back to a page you can open beside the chat.

This is a trimmed-down reference version of a project I built during an
internship. Company-specific data, integrations and deployment pieces have been
removed. What remains is the core of the app and the minimum it needs to run.

## How it works

- **Machine-scoped retrieval.** Documents live in one folder per machine
  (`IMM <number> - <MAKE> (<model>) <serial>`), plus a shared `_general`
  folder. A question only ever searches the selected machine's documents and
  the shared library.
- **Hybrid search.** Vector similarity and an exact-match keyword pass (alarm
  codes, part numbers, oil grades) are merged with reciprocal rank fusion,
  then diversified so one long manual cannot crowd out a one-page datasheet.
- **Progressive retrieval with an evidence judge.** A fresh question runs a
  multi-pass search. A separate model call judges whether the excerpts actually
  answer the question and, if not, widens the search with a better query.
- **Tool-calling answer loop.** The answering model (OpenAI Responses API)
  decides whether to search again, read more of a cited page, or look at a
  rendered page image. After a full search it must verify the value it is
  about to state against the page image, because spec tables often lose their
  row structure in text extraction.
- **Machine identity.** The model is told which machine it is answering about
  (number, model, curated facts from `app/machines/machine_facts.json`) so it
  can pick the right row out of tables that cover a whole product line.
- **Streaming UI.** Progress, the model's reasoning summary and the answer
  stream to the browser over SSE. Citations open a page preview, and full
  answers come with a short summary and a "More detail" toggle.
- **Small extras.** Login with PBKDF2-hashed passwords and rate limiting, chat
  history per machine, thumbs up/down feedback, and a hint when a similar
  question was answered before.

Everything is stored in one SQLite file (chunk index, users, chat history and
retrieval diagnostics).

## Layout

```
app/
  api/        FastAPI routes: auth, machine registry, chat (SSE), previews
  auth/       session login gate, password hashing, rate limiting
  db/         SQLite schema, chat history, feedback, diagnostics
  machines/   machine registry built from folder names, curated facts
  rag/        parsing, chunking, embeddings, hybrid retrieval, judge, tool loop
web/          single-page technician UI (vanilla JS, no build step)
scripts/      index_documents.py, create_user.py
tests/        offline end-to-end smoke test
```

## Run it

Needs Python 3.11+ and an OpenAI API key (or Azure OpenAI, see `.env.example`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # set OPENAI_API_KEY and SESSION_SECRET
```

Put documents (PDF, XLSX, PPTX) under `data/documents/`:

```
data/documents/
  IMM 101 - ACME (PX 200-50) SN12345/
    Operating Manual.pdf
  _general/
    Molding Defects.pdf
```

Index them, create a login, and start the app:

```bash
python scripts/index_documents.py
python scripts/create_user.py tech1 --display-name "Tech One"
uvicorn app.main:app --port 8000
```

Open http://localhost:8000/ and sign in with the printed password.

Or with Docker: `docker build -t imm-assistant .` and run it with the same
environment variables and `data/` mounted at `/app/data`.

## Tests

```bash
pytest
```

The smoke test builds a small document tree, indexes it with stand-in
embeddings, and drives a full chat turn through the real API with a scripted
stand-in for the model. It makes no network calls.
