# Learn EvidenceOps by following one request

Start with `main.py`, then `api.py`, `rag.py`, `retrieval.py`, `ollama_client.py`, `steel_faults.py`, `data_io.py`, `database.py`, `investigation.py`, `mcp_server.py`, `mcp_client.py`, `evaluation.py`, `workflow.py`, and `postgres_smoke.py`. Read the HTML in `dashboard.html` after you know the API names. Run `.venv/bin/python -m unittest -q` as you change code.

## The simplest mental model

Mode A is a **made-up Line B investigation** for practicing the workflow. Mode B is **real UCI Steel Plates Faults data** for exact, reproducible counts. They do not describe the same factory. The application sends only a small, calculated Mode B summary to the language model; Python counts the original rows.

| File | Why it exists; what runs first | Main input → output | Calls and errors | Try this |
|---|---|---|---|---|
| `main.py` | Terminal entry point. `main()` reads options, then `run_command()` chooses a command. | `create_demo_data(folder)` → number of files; `command_steel(args)` → printed counts. | Calls data loaders, search and RAG; top-level catches file, value, model and database errors. | Run `python main.py steel`. |
| `api.py` | Browser entry point. FastAPI matches a URL to a function after `request_limits()`. | `ask(payload)` → verified answer dictionary; `steel_facts()` → computed summary. | Calls `rag.py` or `steel_faults.py`; rejects missing tokens and invalid requests. | Open `/api/health`. |
| `rag.py` | Connects search, Ollama and saved runs. | `answer_question(question, method, documents)` → cited answer; `run_agent(request, allow_draft)` → bounded trace. | Calls retrieval, Ollama, SQLite and MCP; model and storage errors propagate to the API. | Trace `answer_question()` with a keyword question. |
| `retrieval.py` | Four small ranking methods. | `search_bilingual(documents, question)` → scored chunks; `search_bm25(...)` → BM25 ranked chunks; `search_vectors(...)` → cosine ranked chunks; `hybrid_search(...)` → combined ranking. | Takes ordinary dictionaries; checks bad vectors and empty queries. | Search for `calibration records`. |
| `ollama_client.py` | Local model requests and answer checks. | `embed_texts(texts, ...)` → vectors; `grounded_answer(question, evidence, config)` → checked claims. | Uses the loopback Ollama HTTP API; handles connection, malformed output and unsupported claims. | Read `check_claims()` first. |
| `steel_faults.py` | Downloads and calculates real UCI facts. | `load_dataset(folder)` → names and numeric rows; `summarize(dataset)` → counts; `steel_documents(folder)` → source-linked evidence chunks. | Reads fixed-hash raw files; rejects unexpected format, labels and non-finite numbers. | Sum all seven category counts. |
| `data_io.py` | Reads and versions synthetic documents and CSV files. | `load_documents(...)` → chunks; `ingest_document(...)` → saved version; `read_csv(path, required)` → row dictionaries. | Uses files and SQLite; checks names, hashes, size, version history and PDF extraction. | Add one text version in the dashboard. |
| `database.py` | Persists vectors and AI runs; also contains optional PostgreSQL helpers. | `embedding_cache(path, model_key)` → saved vectors; `save_ai_run(path, result)` → run ID. | Uses SQL transactions and locks where needed. Database errors are not ignored. | Restart the server and read saved runs. |
| `investigation.py` | Computes synthetic metrics and manages incident states. | `summarize_metrics(rows)` → numeric summary; `prepare_ticket(...)` → pending draft; `change_ticket(...)` → new state. | Reads synthetic CSV, writes SQLite, checks state transitions. | Try approving without the approval token. |
| `mcp_server.py` | Exposes exactly four approved tools to the agent process. | `search_approved_documents(question)` → SOP chunks; `prepare_incident_ticket(question)` → pending ticket. | Calls document and investigation functions; checks viewer/operator role. | List the four `@server.tool()` functions. |
| `mcp_client.py` | Small example of a real MCP client. `main()` opens a stdio session. | MCP tool request → tool result. | Talks to the MCP server; connection errors end the example. | Run the client with the documented tokens. |
| `evaluation.py` | Offline retrieval checks and real-model case runner. | `run_ai_evaluation(local_dir)` → report and JSON file. | Calls RAG and agent; records failures without inventing human labels. | Compare `regression`, `heldout`, `injection`, `steel` results. |
| `workflow.py` | Fixed two-step synthetic workflow, separate from the AI agent. | `run_workflow(question, kind)` → trace and report. | Calls search, calculation and optional ticket drafting. | Run `python main.py workflow --kind inspection`. |
| `postgres_smoke.py` | Optional PostgreSQL and pgvector verification. | `run_checks(dsn)` → pass/fail result. | Uses `database.py`; needs a running PostgreSQL service. | Read it after SQLite makes sense. |
| `test_evidenceops.py` | Tests behavior and security boundaries. | `unittest` methods → pass, fail or skip. | Uses tiny fixtures and mocks; eight tests need the unavailable starter dataset. | Break one assertion, see it fail, restore it. |

## One question, line by line

1. The browser's **Generate grounded answer** button posts `{question, method, dataset}` to `/api/ask`. The terminal alternative is `python main.py ask --question ...`.
2. `api.ask()` checks the access token. `chosen_documents()` picks either synthetic evidence or real UCI summaries. It never combines them.
3. `rag.answer_question()` obtains model settings, then calls `retrieve()`.
4. `retrieve()` calls `retrieval.py` for keyword or BM25 search. Semantic and hybrid methods call `build_index()` first. That function asks Ollama for embeddings only when a matching cached vector is absent.
5. The best chunks become the evidence list. `ollama_client.grounded_answer()` removes suspected instructions inside source text, asks `qwen2.5:7b` for claims, and calls `check_claims()`.
6. `check_claims()` checks that each cited ID exists and each quoted span occurs exactly in that passage. `claim_supported()` asks the model whether the claim follows from only that passage. Invalid claims are withheld; empty claims abstain.
7. `database.save_ai_run()` saves the answer, evidence, timings and checks. `api.ask()` sends that dictionary back to the dashboard, which displays citations and rejected claims.

**Read `rag.answer_question()` closely:** `config` holds the model names. `documents` is a list of evidence dictionaries. `retrieval` holds scores and timings. The `for row` loop copies only the fields needed for cited evidence. `grounded_answer()` returns checked claims. The final assignments add the question, method and timings. `save_ai_run()` returns a run ID, which the function includes in its returned dictionary. These ordinary assignments are the full answer path.

## Agent and human boundary

`rag.run_agent()` starts a real MCP server process through `server_settings()`. That function removes the approval secret from the child environment. `available_actions()` offers only the four approved tools within budgets. A search must happen before drafting or finishing; `tool_arguments()` rejects a call that was not allowed. `call_tools()` records each step and prevents repeats. `mcp_server.py` checks the role again. A draft is always **pending**. The separate `/api/tickets/{id}/approve` route checks the human approval token and writes an audit event. The agent has no approval tool.

MCP requires asynchronous session code because the protocol uses a live stdio conversation. FastAPI route decorators and SQLite transactions also remain because they implement routing and safe state changes. Their syntax is worth learning after the simple search and counting functions.

## Real UCI data path

1. `python main.py steel-download` fetches the fixed UCI archive and verifies the SHA-256 of `Faults.NNA` and `Faults27x7_var`.
2. `load_dataset()` reads original names and 1,941 tab-separated rows. `read_rows()` converts each value to a finite number.
3. `fault_of()` verifies exactly one of seven labels equals 1 per row. `summarize()` loops over the rows, counts labels, and calculates feature statistics.
4. `/api/steel` returns these deterministic numbers. The **Real steel data** tab displays them with the source and license.
5. `summary_sentences()` creates small, attributed evidence chunks. Choosing Mode B in **AI answers** searches only these chunks. The model never counts the raw rows.

No normal-production denominator exists, so these data cannot answer a factory-wide defect-rate question.

## Exercises in order

1. **Variables:** print the `records` value returned by `summarize()`.
2. **Loops:** write a loop that prints each fault label.
3. **Lists:** collect names of categories with more than 200 records.
4. **Dictionaries:** build `{name: count}` from the category list.
5. **Functions:** write `largest_category(categories)` with a loop.
6. **File handling:** read the first line of `Faults27x7_var`.
7. **CSV:** add a synthetic metrics row and inspect the report.
8. **Search:** trace how `search_documents()` scores one chunk.
9. **Cosine:** calculate cosine similarity for `[1, 0]` and `[0, 1]` by hand.
10. **APIs:** call `/api/steel` with the access token using `curl`.
11. **RAG:** explain why a citation ID alone is insufficient without a matching quote.
12. **MCP:** identify the check that stops an unauthorized tool name.
