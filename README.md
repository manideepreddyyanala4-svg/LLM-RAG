# EvidenceOps — local development

EvidenceOps is a learning project for inspecting synthetic industrial records,
searching cited evidence, calculating investigation facts, and preparing incident
drafts. It is **not production ready**. No model response or benchmark label is
invented. The source data remains local and is excluded from Git.

The local flow is: terminal or browser → document loader → lexical search and
deterministic calculations → cited results. SQLite stores uploaded documents,
version events, label reviews, and ticket decisions. MCP tools expose only
selected data operations. PostgreSQL and model adapters are optional paths.
The three fixed investigations use ordinary Python functions; no graph runtime
or model is needed for them.

## Quick start in Windows CMD

From the existing project folder:

```cmd
cd /d "D:\LLM-RAG project"
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest -q
set EVIDENCEOPS_TOKEN=choose-a-long-random-local-secret
set EVIDENCEOPS_OPERATOR_TOKEN=choose-a-different-long-random-operator-secret
set EVIDENCEOPS_APPROVAL_TOKEN=choose-a-different-long-random-secret
.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` in the same laptop's browser. Enter the three
distinct tokens in its header. The access token reads evidence; the operator
token manages local documents and prepares incident drafts; the approval token
confirms manual labels and changes ticket decisions. The access token is also
required for operator and approver requests. These values stay in your CMD
session and are not saved by the dashboard. Use a private laptop session and
do not share them. Token rotation takes effect on the next request. This is a
local role boundary, not user identity or production authentication.

The terminal application remains available:

```cmd
.venv\Scripts\python.exe main.py
.venv\Scripts\python.exe main.py inspect
.venv\Scripts\python.exe main.py investigate
.venv\Scripts\python.exe main.py labels
.venv\Scripts\python.exe main.py workflow --kind inspection --question "Line B rejection rate"
.venv\Scripts\python.exe main.py workflow --kind deployment --question "Compare Line B deployments"
.venv\Scripts\python.exe main.py workflow --kind sop_conflict --question "Review Line B SOP versions"
.venv\Scripts\python.exe main.py backup
```

The SOP workflow creates a **pending** ticket; approval remains a separate
action. The backup command copies existing SQLite stores into a new timestamped
folder under ignored `data/local/backups/`. It never overwrites a prior backup.
Restore writes only into a fresh local directory and refuses existing database
files. For an isolated local run, set `EVIDENCEOPS_LOCAL_DIR` to an absolute
path before starting the program. Leave it unset for the usual `data/local`
stores.

On this laptop, `.venv\Scripts\python.exe -m unittest -q` ran 53 tests and
passed. Isolated headless Chrome exercised overview, pasted and file upload,
current/historical search and citations, soft deletion and restore, document
history, Parquet metadata, investigations, deployment comparison, SOP workflow,
workflow history, operator/approver denial and success, review screen, 20
comparison rows, and the unreviewed evaluation gate. Uploaded script text was
displayed without running. The JSON download control was exercised with an
explicit toy browser response; the actual reviewed-only export is tested with
a separate synthetic fixture. No supplied question was automatically reviewed.
Temporary browser profiles and databases were removed afterward. A later
isolated Chrome run also downloaded and checked the reviewed JSON for one
separate toy question. No supplied starter question was marked reviewed.

## Fresh clone on the authorized lab computer

The original starter files are intentionally absent from Git because their
README gives no explicit redistribution license. The public Parquet dataset is
also absent. For a first run, `python main.py demo` creates six **new, tiny,
explicitly synthetic** files in `EvidenceOps_Data_Starter/data/`: three SOPs,
metrics, deployments, and one unreviewed practice question. It refuses to
overwrite any of those names. The practice values are not factory findings or
a retrieval benchmark. Keep actual lab records out of Git.

Windows Command Prompt (use an installed Python 3.14 and Git):

```cmd
git clone https://github.com/manideepreddyyanala4-svg/LLM-RAG.git
cd LLM-RAG
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py demo
.venv\Scripts\python.exe -m unittest -q test_evidenceops.EvidenceTests.test_demo_data_is_explicitly_synthetic_and_never_overwrites
set EVIDENCEOPS_TOKEN=choose-a-long-random-local-secret
set EVIDENCEOPS_OPERATOR_TOKEN=choose-a-different-long-random-operator-secret
set EVIDENCEOPS_APPROVAL_TOKEN=choose-a-different-long-random-secret
.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Linux shell (use an installed Python 3.14 and Git):

```sh
git clone https://github.com/manideepreddyyanala4-svg/LLM-RAG.git
cd LLM-RAG
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py demo
.venv/bin/python -m unittest -q test_evidenceops.EvidenceTests.test_demo_data_is_explicitly_synthetic_and_never_overwrites
export EVIDENCEOPS_TOKEN='choose-a-long-random-local-secret'
export EVIDENCEOPS_OPERATOR_TOKEN='choose-a-different-long-random-operator-secret'
export EVIDENCEOPS_APPROVAL_TOKEN='choose-a-different-long-random-secret'
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` on that computer. The full 53-test suite has
checks tied to the original ignored starter data and will fail on the tiny
demo; run it only after placing an authorized copy of the original starter
files there. GitHub Actions runs its smaller, dataset-free smoke selection.
For Docker, create `data/public/` as an empty local directory if you have no
permitted Parquet files, set the three tokens above, then run
`docker compose -f compose.yaml up --build`. If PostgreSQL/pgvector is needed,
also set a private `EVIDENCEOPS_PG_PASSWORD` and use
`docker compose -f compose.yaml --profile postgres up --build`.
The database profile is optional; the normal API continues to use SQLite.

An offline MCP client/server round trip (using the same first token):

```cmd
.venv\Scripts\python.exe mcp_client.py
```

For a fresh installation, use Python 3.14, create `.venv` with `python -m venv
.venv`, then install requirements. Place authorized local copies of the three
SOP texts, `metrics.csv`, `deployments.csv`, and `questions.csv` in
`EvidenceOps_Data_Starter/data/`. Place the two public Parquet files in
`data/public/` if you want Parquet inspection. The application never downloads
datasets. Empty or malformed required inputs produce errors instead of made-up
answers. `data/local/` holds ingested text versions and incident drafts.

## What each file teaches

Read in this order:

1. `retrieval.py`: `words(text)` extracts English terms; `chinese_pairs(text)`
   extracts adjacent Han characters; `search_documents(documents, question)`
   keeps the original baseline; `search_bilingual(documents, question)` returns
   up to three scored chunks. `search_bm25` offers frequency and length based
   lexical ranking. `search_vectors` ranks vectors supplied by a caller and
   rejects invalid dimensions, values, and zero-length vectors; it does not
   create embeddings.
   `hybrid_search` accepts a semantic search function later and rejects
   evidence that was not supplied to it.
2. `data_io.py`: `read_csv(path, required)` returns validated dictionaries;
   `load_documents(folder, include_csv, local_file)` returns text, filename,
   location, version, status, and SHA-256 where applicable;
   `ingest_document(store, source, version, text)` and `ingest_pdf` add a new
   SQLite version with original bytes, a hash, and paragraph or page citations;
   `delete_document(store, source, version)` soft-deletes one local version;
   `restore_document` makes a deleted version current and records the change;
   `document_history` reads its events; `backup_local_databases` makes a
   checked SQLite snapshot and `restore_local_databases` restores into a fresh
   directory without overwriting;
   `document_inventory(documents)` shows version hashes and change flags;
   `inspect_parquet(folder)` prints metadata and five sample rows.
3. `investigation.py`: `summarize_metrics(rows)` checks counts and computes
   line/model rates; `investigation_report(folder, documents)` returns facts
   with source records; `prepare_ticket`, `list_tickets`, `change_ticket`, and
   `ticket_history` manage persistent decisions and the audit trail.
4. `evaluation.py`: `evaluate_questions(documents, questions, verified, search)`
   returns top-1 and top-3 source hits for reviewed questions;
   `review_label` saves a source review tied to the current CSV and cited-file hashes;
   `compare_searches` records the two lexical ranks and timings.
5. `main.py`: `main()` handles the original terminal commands and includes
   current ingested local documents in search and comparisons. `evaluate
   --json` prints a reviewed-only report for saving or inspection.
6. `api.py` and `dashboard.html`: local HTTP endpoints and responsive browser
   controls for search, citations, document versions, Parquet metadata, reports,
   label review, incident drafts, approval, and fixed workflow traces.
7. `mcp_server.py` and `mcp_client.py`: official MCP SDK stdio server and client.
   Four tools search approved local texts, read metrics, compare deployment
   records, and prepare a draft. The server requires a configured access token
   in its environment. Ticket preparation also requires a separate operator
   token and `EVIDENCEOPS_MCP_ROLE=operator`; its default role is viewer.
   The client removes the approval token from the child environment and also
   removes the operator token in viewer mode. Tool arguments never include
   secrets. This is a process-wide role, not
   per-client identity; only trusted local processes should launch it.
   There is no ticket-approval tool.
8. `workflow.py`: `run_workflow(question, kind)` executes bounded inspection,
   deployment, or SOP-version review steps. Reports list sources and missing
   information. `record_workflow_run` and `list_workflow_runs` keep the latest
   50 completed API run reports in SQLite. Internal file paths stay out of
   responses. The SOP workflow prepares a pending ticket. No LLM reasoning
   or automatic ticket approval is claimed.
9. `database.py`: optional parameterized PostgreSQL schema and functions for
   versioned documents, chunks, inspection rows and inactive-version deletion.
   `setup_vectors(dsn, dimensions)` requires an available pgvector extension
   and a real embedding dimension. Nothing connects without an explicit DSN.
10. `ollama_client.py`: `list_ollama_models`, `ask_ollama`, and `embed_texts` require explicit approval
    for later lab requests; `build_grounded_prompt` formats real citations
    without calling a model. The current UI and API do not call a model.
    `test_evidenceops.py` checks the offline behavior.
11. `postgres_smoke.py`: `add_version` writes one synthetic document version;
    the three `check_*` functions test versions, concurrency, inspection rows,
    and synthetic vectors; `run_checks` creates a temporary test schema, runs
    them, then removes only that schema. Set `EVIDENCEOPS_TEST_DATABASE_URL`
    explicitly before running it.

## Local documents and backup

The Documents tab accepts pasted text or a `.txt`/`.pdf` upload with the access
and operator tokens. UTF-8 text is limited to 100 KB. PDFs are limited to 2 MB
and 20 pages; extracted text is cited by page. Uploaded bytes are retained in
SQLite and can be downloaded with the access token until the version is deleted.
Creating a new version preserves the old one. Restoring a deleted local version
requires the operator token, checks its stored-file hash, and supersedes any
current version of that source. A failed integrity check leaves the current
version in place.
Creation, superseding, deletion, restoration,
and legacy import events appear in document history. Existing versions from
before the byte-storage migration still have searchable text but may not have
their original downloadable file. PDF tables may be flattened into text.
For a page without an extractable text layer, the app optionally runs Poppler's
`pdftoppm` and Tesseract from `PATH`. Install both separately if scanned PDFs
are required; no Python OCR package is needed. On Windows CMD, check with
`where pdftoppm` and `where tesseract`. The default language is `eng`. To add
Traditional Chinese after installing its Tesseract language data, use
`set EVIDENCEOPS_OCR_LANG=eng+chi_tra` before starting the API. OCR has a
60-second total deadline and per-command timeouts. Missing tools, language
data, or empty recognition reject the entire PDF; no empty evidence is saved.
OCR execution was tested only with mocks here because both tools are absent.
Only trusted local operators should upload PDFs; bounded input size does not
make arbitrary PDFs safe for a public service.

To restore a backup, stop the API/MCP processes, inspect the backup folder,
and select a new empty local directory. In CMD:

```cmd
set EVIDENCEOPS_LOCAL_DIR=D:\LLM-RAG project\data\local\restored-copy
.venv\Scripts\python.exe main.py restore --backup-dir "D:\LLM-RAG project\data\local\backups\YOUR_TIMESTAMP"
```

The command checks every backup database, restores `documents.db`, `tickets.db`,
and `reviews.db` when present, and refuses to overwrite current stores. Start
the API in that same CMD session to use the restored copy. Keep the original
stores until you have checked the restored records.

## Actual local observations and limits

The three SOP texts, `metrics.csv`, `deployments.csv`, and `questions.csv` are
the original synthetic educational examples described in the locally supplied
`EvidenceOps_Data_Starter/data/README.txt`. That file states that they are not
real factory or laboratory records. It does not give a separate redistribution
license for the starter bundle, so confirm publication rights before sharing it.
The two public Parquet files came from Jeremy Vachier's
[Industrial Maintenance Synthetic Dataset](https://huggingface.co/datasets/Jvachier/industrial-maintenance-synthetic).
Its local dataset card and upstream page state an Apache 2.0 license. Keep the
upstream attribution and license notice if you redistribute those data. Both
datasets remain excluded from this repository.

The starter data has 3 SOP files, 8 metric rows, 3 deployment rows and 10
questions. The SOPs become 6 blank-line paragraphs; with optional CSV evidence
there are 17 searchable chunks. The maintenance Parquet file has 1,048,779
rows and 20 columns; sensor tags have 1,048,779 rows and 9 columns. Parquet
inspection uses metadata and a batch of at most five rows.

Line B v3 rejection: 61/2000 = 3.05%. Line B v4: 145/2000 = 7.25%, a
4.20 percentage point rise. Line A v3: 118/4000 = 2.95%. These are rejection
rates, not false-rejection rates. They do not establish cause. The Line B v4
deployment date (2026-10-03) and active SOP date (2026-10-04) differ.

The supplied `questions.csv` has `question_id`, `question`, `expected_answer`,
`expected_source`, and `question_type`, but no recorded human verification.
In the Evaluation tab, choose **Review labels**. For each of the 10 questions,
the supplied answer and sources are shown as unverified proposals. Choose a
source, click **Show source evidence**, read the cited paragraph or record,
then type the source filename(s) you actually confirmed. The confirmed field
starts blank. Check the inspection box and save using the separate approval
token. The server rejects review submissions without explicit confirmation.
You can also use `main.py labels` and open the source files yourself. Only
saved reviews tied to the current `questions.csv` and unchanged cited source
files can be evaluated. Older reviews without source hashes require a fresh
manual review:

```cmd
.venv\Scripts\python.exe main.py evaluate --include-csv
.venv\Scripts\python.exe main.py evaluate --include-csv --json
.venv\Scripts\python.exe main.py compare --include-csv
```

The comparison command writes raw ranks, scores, timings, and SHA-256 hashes
of the exact evidence chunks and question texts to
`data/local/retrieval_experiments.jsonl`. It marks top-1 and top-3 misses only
for saved reviews. Unreviewed rows are unscored; an empty result is explicitly
shown as having no lexical matches. Top-3 means **any expected source in
the first three chunks**; it is not full multi-source recall or answer quality.
Earlier 4/10 top-1 and 5/10 top-3 results were unverified-label diagnostics
for SOP-only search, not a verified benchmark. No new verified result is claimed.
The dashboard also offers a JSON download after a reviewed evaluation succeeds.

API and MCP search use current/active evidence by default. API search can
narrow results to an exact source filename or version. Set `active_only` to
`false` in the API or `include_history` to `true` in MCP to inspect a
superseded SOP or local upload deliberately. Deleted uploads remain excluded
from search even in historical mode. Evaluation still compares historical sources
where the reviewed question calls for them. Operational investigations and
ticket drafts cite current SOPs; the SOP comparison explicitly reads history.
Chinese
search uses character pairs, not reliable word segmentation. Metadata
identifies superseded SOPs; equal scores favor active documents. Hashes detect
exact duplicates and changed content across local versions. There is no live
source-watch process, embedding model, vector index, or model-backed semantic
retrieval. The offline vector function uses supplied numeric test vectors only.
Search results are evidence candidates; an adversarial document is displayed as
data and cannot change ticket state by itself. Local documents use SQLite
transactions and retain old and deleted versions for inventory; search reads
only current versions. Earlier `documents.json` data is imported once if present.

## Verified development progress

- The original 14 tests passed before changes.
- The expanded suite now has 53 passing tests, including ticket restart,
  migration, concurrent decisions, local document concurrency, soft deletion,
  PDF page citations and byte-hash checks, API access failures, MCP tool
  failures and timeout handling, malformed CSV and legacy data, workflow
  calculations, SQLite backup, partial label review, vector validation, and
  request-key conflicts, exact uploaded text bytes, and source-change review
  invalidation. Offline vector checks also reject zero embeddings and invalid
  database dimensions. Soft-delete restoration, filtered search, and approval
  denial were also verified. Workflow run history survives a restart and
  reviewed-only JSON export refuses unreviewed labels.
- The source-level audit fixed historical search for superseded local uploads,
  removed absolute paths from API/MCP backend errors, hashes starter SOP bytes
  exactly, rejects forged review-store paths, and removes partial SQLite
  backups after failure. The fixed two-step workflow no longer needs LangGraph.
  The optional PostgreSQL smoke test now forces a SQL constraint failure and
  verifies the prior active version survives transaction rollback.
- Successful review and JSON evaluation are exercised with an explicitly
  constructed toy question and document in a temporary directory. Automated
  tests do not confirm any of the ten supplied starter questions.
- An isolated full scenario ingested actual starter SOP text, searched its
  citation, calculated inspection and deployment findings, prepared a ticket,
  denied missing and wrong approval tokens, approved with the separate token,
  reopened persisted tickets and workflow runs, rejected a rotated access token,
  and retrieved the local copy through the real MCP protocol.
- The 10 starter question texts produced 20 keyword/BM25 runs with actual
  source ranks and timings. These labels are unreviewed, so no accuracy
  improvement is claimed. Q01's first result differed: keyword ranked
  `sop_v2_en.txt`, while BM25 ranked `deployments.csv` in this run.
- A real isolated headless Chrome check passed for the dashboard interactions
  listed above. A first browser-helper attempt failed; a later correctly quoted
  isolated Chrome profile provided a working debug port. Visual inspection on
  your display and hosted-browser CI have not run.
- A later headless Chrome audit exercised historical local-upload search,
  deletion exclusion, deployment workflow display, and the unreviewed evaluation
  message. At a 390-pixel viewport, the document width stayed at 390 pixels;
  the navigation row scrolls horizontally within the page. The mobile
  screenshot was inspected locally. A separate toy review screen saved one
  manually confirmed source, showed a 1/1 evaluation, and downloaded a valid
  reviewed-only JSON file in real Chrome. This is a browser-function check,
  not a benchmark score for the starter questions.
- A local MCP stdio client/server round trip discovered four tools and returned
  a real cited SOP passage for `calibration records`. A separate isolated stdio
  check called all four tools: search, Line B inspection metrics, and deployment
  records succeeded; viewer-mode ticket preparation and an invalid line were
  denied; operator mode prepared a pending draft. The container's MCP client
  also discovered four tools and retrieved cited SOP evidence. No MCP tool can
  approve a ticket.
- Local API requests have a per-address rate limit and a body-size
  limit of 200 KB, checked against actual bytes even if the declared length
  is wrong. Requests with bodies need `Content-Length`. The dashboard response
  disallows framing and remote script loading. These local
  controls are not a substitute for a production gateway or user accounts.

The optional PostgreSQL adapter ran against a local PostgreSQL 17 container
with pgvector 0.8.7. It created the schema and vector column, stored two
synthetic document versions, rejected a changed hash for an existing version,
refused to delete the active version, deleted the inactive version, and stored
an inspection row twice without duplication. A conflicting inspection count
was rejected. A cosine-distance SQL query returned the expected row for
**manually supplied synthetic vectors**. A dimension mismatch was rejected,
and the active document and inspection row survived a database-container
restart. A separate race check found 7/10 simultaneous version pairs with two
active versions before repair. A per-source transaction lock fixed that: 20
competing pairs and four concurrent writes of one version passed afterward.
The checked-in smoke script also passed 10 independent competing pairs and a
repeatable legacy-schema migration. It also passed against a newly created
database with no prior pgvector extension. This tests the optional adapter,
not model embeddings. The normal API still uses SQLite and does not switch to
PostgreSQL automatically.
The fixed two-step Python workflow and MCP discovery/execution were tested locally.
The AI agent endpoint returns 503 until real model inference is connected.
No live Ollama request, real embeddings, model-backed retrieval, agent
reasoning, or lab integration has been tested. Ollama generation and embedding
response validation was tested with mocked HTTP responses only. Human approval
here is a separate local token and
an explicit state transition; it is not an enterprise identity system.

## Lab Ollama connection, later

1. On the lab machine, confirm that Ollama is already running and keep it bound
   to the local computer. Do not expose port 11434 publicly. Confirm which
   evidence may be sent to it. Discover installed models; choose the embedding
   and generator names from the real response. Do not assume either is installed.

   ```cmd
   curl.exe http://127.0.0.1:11434/api/tags
   ```

   On Linux, use `ollama list` or `curl http://127.0.0.1:11434/api/tags`.
   The Python `list_ollama_models(base_url, approved=True)` adapter can also
   list names and digests; it has only been tested with mocked responses here.

2. Request a test embedding using the approved embedding model and verify that
   `embeddings` contains one nonempty numeric vector. Record its dimension.
   Use the same model and dimension for every current document chunk and every
   question. If a source hash or model changes, rebuild its vectors.

   ```cmd
   curl.exe http://127.0.0.1:11434/api/embed -H "Content-Type: application/json" -d "{\"model\":\"YOUR_EMBED_MODEL\",\"input\":\"test passage\",\"truncate\":false}"
   ```

   On Linux, use `curl http://127.0.0.1:11434/api/embed -H 'Content-Type: application/json' -d '{"model":"YOUR_EMBED_MODEL","input":"test passage","truncate":false}'`.

3. Request one nonstreaming generation with the approved generator model.
   Check that the response has text and `done: true`:

   ```cmd
   curl.exe http://127.0.0.1:11434/api/generate -H "Content-Type: application/json" -d "{\"model\":\"YOUR_GENERATOR_MODEL\",\"prompt\":\"Say hello\",\"stream\":false}"
   ```

   On Linux, use `curl http://127.0.0.1:11434/api/generate -H 'Content-Type: application/json' -d '{"model":"YOUR_GENERATOR_MODEL","prompt":"Say hello","stream":false}'`.

4. On the authorized lab machine, call `embed_texts(texts, base_url, model,
   approved=True)` for current cited chunks and questions. Check that model
   name, dimension, source version, and content hash stay consistent. Feed the
   vectors in document order to `search_vectors(documents, query_vector,
   document_vectors)` and compare against **manually reviewed** questions.
   Use `build_grounded_prompt(question, evidence)` and call `ask_ollama(...,
   approved=True)` only there. Test cited answers and tool permissions before
   enabling `/api/agent`; the laptop endpoint still returns 503.

The remaining **code** work on the lab machine is to save real chunk vectors
with `(source, version, location, source SHA-256, embedding model, model digest,
dimension)`, invalidate them when any of those change, embed each question
with the same model, and connect `search_vectors` or `hybrid_search` to that
store. `database.py` has a pgvector column but no model metadata or
production vector ingestion/search path. Next, use only retrieved chunk IDs
to build the prompt, call the generator, and validate every returned citation
against those IDs; an unsupported citation must be rejected. A bounded agent
may then call only the existing approved MCP tools, with no approval tool and
no automatic ticket decision. Record actual retrieval ranks, model latency,
answer quality against human reviewed questions, and failure cases before
claiming RAG or agent performance. These integrations are **not implemented**
by the current API.

Ollama API references: [models](https://docs.ollama.com/api/tags),
[embeddings](https://docs.ollama.com/api/embed), and
[generation](https://docs.ollama.com/api/generate). The current adapter uses a
10-second per-operation timeout, a 1 MB generation response cap, a 2 MB
embedding response cap, and no redirects or
proxies. It does not enforce a whole-request deadline; a real lab integration
must measure end-to-end time and decide a suitable limit.

## Packaging and GitHub status

`requirements.txt` pins the runtime versions; LangGraph was removed after the
two-step workflow was simplified. `Dockerfile` and
`.dockerignore` prepare an optional container. `compose.yaml` mounts starter
data read-only, persists `data/local`, requires three distinct tokens, and
exposes only `127.0.0.1:8000` on the host by default. In Windows CMD:

```cmd
set EVIDENCEOPS_TOKEN=choose-a-long-random-local-secret
set EVIDENCEOPS_OPERATOR_TOKEN=choose-a-different-long-random-operator-secret
set EVIDENCEOPS_APPROVAL_TOKEN=choose-a-different-long-random-secret
docker compose -f compose.yaml config --quiet
docker compose -f compose.yaml up --build
```

Docker Desktop started locally, the image built, and the app container became
healthy. In an isolated Compose project bound to host port 8768, the real HTTP
API denied a request without an access token, ingested and searched a test
document, denied ticket approval without the approver token, and approved the
ticket with it. The document and ticket survived an app-container restart.
The container also returned a cited search hit, ran inspection, deployment,
and SOP workflows, kept three workflow reports after a second restart, and
returned 20 raw comparison rows. Reviewed-only evaluation correctly returned
409 because no human label reviews were saved in that isolated store. An
immediate request during restart failed while the server was starting; the
health check became ready and the persistence check then passed.
Inspection of the image environment found none of the test tokens baked in.
The test project used its own ignored local data directory; the usual stores
were untouched. The host-port and mount overrides are for isolated testing;
the defaults remain 8000 and `data/local`.

For the optional PostgreSQL/pgvector service, set a private password in the
same CMD session and start the profile. The database has no published host
port; it is available to containers on this Compose network as `db:5432`.
An empty password prevents the database image from initializing. Keep the
password stable for an existing named volume. `database.py` connects only
when a caller explicitly supplies a DSN.

```cmd
set EVIDENCEOPS_PG_PASSWORD=choose-a-private-local-database-password
docker compose -f compose.yaml --profile postgres config --quiet
docker compose -f compose.yaml --profile postgres up --build
docker compose -f compose.yaml --profile postgres ps
```

To repeat the database smoke check in the app container, set a DSN for this
**test**, with a URL-encoded password if needed. It creates a new random schema
and removes that schema afterward; use an isolated database if possible.

```cmd
set "EVIDENCEOPS_TEST_DATABASE_URL=postgresql://evidenceops:YOUR_URL_ENCODED_PASSWORD@db:5432/evidenceops"
docker compose -f compose.yaml --profile postgres exec -e EVIDENCEOPS_TEST_DATABASE_URL evidenceops python postgres_smoke.py
```

The app image does not include optional OCR executables, so scanned PDFs are
rejected there unless a separately reviewed OCR image is prepared. Tests need
mounted real fixtures. `.github/workflows/offline.yml` defines 30 dataset-free
smoke tests, source compilation, and a separate PostgreSQL/pgvector service job
for future GitHub CI. Its YAML parsed; all 30 selected tests and the equivalent
live database smoke script passed locally. No hosted CI run has occurred. The
full 53-test suite uses the private starter dataset and runs locally.
`.gitignore` excludes datasets, virtual environments, caches, secrets, SQLite
files and archives. Review every future staged file before publishing.
