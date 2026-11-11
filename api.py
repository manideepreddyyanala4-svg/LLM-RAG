"""Local FastAPI interface. Set EVIDENCEOPS_TOKEN and bind to 127.0.0.1."""
import os
import secrets
import logging
import re
import time
from threading import Lock
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from typing import Literal
import sqlite3
from data_io import DATA, LOCAL, ROOT, document_inventory, document_history, ingest_document, ingest_pdf, read_document_file, delete_document, restore_document, load_documents, read_csv
from retrieval import search_bilingual, search_bm25, current_evidence, not_deleted
from evaluation import evaluate_questions, compare_searches, review_label, read_verified_questions, REVIEW_SOURCES
from investigation import investigation_report, prepare_ticket, list_tickets, approve_ticket, change_ticket, ticket_history
from workflow import run_workflow, record_workflow_run, list_workflow_runs
from rag import answer_question, compare_methods, run_agent, ai_status, build_index, evidence_documents, RUNS
from ollama_client import ai_settings
from database import list_ai_runs
from steel_faults import download_dataset, load_dataset, summarize, steel_documents, STEEL_DIR

app = FastAPI(title="EvidenceOps", docs_url=None, redoc_url=None, openapi_url=None)
STORE = LOCAL / "documents.db"
TICKETS = LOCAL / "tickets.db"
REVIEWS = LOCAL / "reviews.db"
RATE_LIMIT = 120
REQUEST_TIMES = {}
RATE_LOCK = Lock()
logger = logging.getLogger("evidenceops")


@app.middleware("http")
async def request_limits(request: Request, call_next):
    """Bound local request rates and log paths/statuses without secrets."""
    started = time.monotonic()
    path = request.url.path
    if path.startswith("/api/"):
        size = request.headers.get("content-length")
        if request.method in ("POST", "PUT", "PATCH", "DELETE") and size is None:
            return JSONResponse({"detail": "Content-Length is required"}, status_code=411)
        if size is None:
            size = "0"
        if path == "/api/documents/upload":
            limit = 2_000_000
        else:
            limit = 200_000
        if not size.isdigit() or int(size) > limit:
            return JSONResponse({"detail": "Request body too large or invalid"}, status_code=413)
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > limit:
                    return JSONResponse({"detail": "Request body too large"}, status_code=413)
                body.extend(chunk)
            # Starlette reuses this cached body when FastAPI parses the request.
            request._body = bytes(body)
        address = "unknown"
        if request.client:
            address = request.client.host
        # Keep only this client's request times from the last 60 seconds.
        with RATE_LOCK:
            recent = []
            for when in REQUEST_TIMES.get(address, []):
                if started - when < 60:
                    recent.append(when)
            if len(recent) >= RATE_LIMIT:
                REQUEST_TIMES[address] = recent
                return JSONResponse({"detail": "Too many requests"}, status_code=429)
            recent.append(started)
            REQUEST_TIMES[address] = recent
    response = await call_next(request)
    elapsed_ms = round((time.monotonic() - started) * 1000, 2)
    logger.info("%s %s status=%s elapsed_ms=%s", request.method, path, response.status_code, elapsed_ms)
    return response


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    include_csv: bool = True
    method: Literal["keyword", "bm25"] = "keyword"
    active_only: bool = True
    source: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,50}\.(txt|pdf|csv)$")
    version: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9._-]{1,40}$")


class DocumentInput(BaseModel):
    source: str = Field(min_length=5, max_length=54)
    version: str = Field(min_length=1, max_length=40)
    text: str = Field(min_length=1, max_length=100000)


class DocumentRef(BaseModel):
    source: str = Field(min_length=5, max_length=54)
    version: str = Field(min_length=1, max_length=40)


class TicketInput(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    request_key: str | None = Field(default=None, max_length=100)


class WorkflowInput(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    kind: Literal["inspection", "deployment", "sop_conflict"] = "inspection"


class AskInput(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    method: Literal["keyword", "bm25", "semantic", "hybrid"] = "hybrid"
    # "synthetic" = Mode A demo SOPs and records; "steel" = Mode B real UCI dataset.
    dataset: Literal["synthetic", "steel"] = "synthetic"


class AgentInput(BaseModel):
    request: str = Field(min_length=1, max_length=1000)


class LabelReview(BaseModel):
    question_id: str = Field(min_length=1, max_length=20)
    expected_source: str = Field(min_length=1, max_length=200)
    confirmed: bool = False


def authorize(x_evidenceops_token: str | None):
    """Require a configured local token for every data endpoint."""
    expected = os.environ.get("EVIDENCEOPS_TOKEN", "")
    if not expected:
        raise HTTPException(503, "Set EVIDENCEOPS_TOKEN before serving data")
    if not x_evidenceops_token or not secrets.compare_digest(x_evidenceops_token, expected):
        raise HTTPException(403, "Invalid access token")


def authorize_approval(x_approval_token: str | None):
    """Require a separate secret that is never passed to MCP tools."""
    expected = os.environ.get("EVIDENCEOPS_APPROVAL_TOKEN", "")
    if not expected:
        raise HTTPException(503, "Set EVIDENCEOPS_APPROVAL_TOKEN to approve tickets")
    if expected in (os.environ.get("EVIDENCEOPS_TOKEN"), os.environ.get("EVIDENCEOPS_OPERATOR_TOKEN")):
        raise HTTPException(503, "Approval token must differ from access and operator tokens")
    if not x_approval_token or not secrets.compare_digest(x_approval_token, expected):
        raise HTTPException(403, "Approval access denied")


def authorize_operator(x_operator_token: str | None):
    """Require the separate operator token for local mutations and drafts."""
    expected = os.environ.get("EVIDENCEOPS_OPERATOR_TOKEN", "")
    if not expected:
        raise HTTPException(503, "Set EVIDENCEOPS_OPERATOR_TOKEN for operator actions")
    if expected in (os.environ.get("EVIDENCEOPS_TOKEN"), os.environ.get("EVIDENCEOPS_APPROVAL_TOKEN")):
        raise HTTPException(503, "Operator token must differ from access and approval tokens")
    if not x_operator_token or not secrets.compare_digest(x_operator_token, expected):
        raise HTTPException(403, "Operator access denied")


def safe_documents(include_csv=False, include_deleted=False):
    """Translate missing or invalid local source files into a structured API error."""
    try:
        return load_documents(DATA, include_csv, STORE, include_deleted)
    except (OSError, ValueError, KeyError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Evidence unavailable") from error


@app.get("/")
def homepage():
    return FileResponse(ROOT / "dashboard.html", headers={
        "Cache-Control": "no-store",
        "Content-Security-Policy": "default-src 'self'; connect-src 'self'; "
        "script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    })


@app.get("/api/health")
def health():
    return {"status": "ok", "data_available": DATA.is_dir(),
            "ai_configured": os.environ.get("EVIDENCEOPS_AI_ENABLED") == "1"}


@app.get("/api/overview")
def overview(x_evidenceops_token: str | None = Header(default=None)):
    """Summarize actual local evidence, reviews, tickets, and model status."""
    authorize(x_evidenceops_token)
    evidence = safe_documents(include_csv=True)
    try:
        questions = read_csv(DATA / "questions.csv", ["question_id", "question"])
        reviewed = read_verified_questions(REVIEWS, DATA / "questions.csv")
        saved_tickets = list_tickets(TICKETS)
    except (OSError, ValueError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Overview unavailable") from error
    pending = 0
    for ticket in saved_tickets:
        if ticket["state"] == "pending":
            pending += 1
    return {"evidence_chunks": len(evidence), "questions": len(questions),
            "reviewed_questions": len(reviewed), "tickets": len(saved_tickets),
            "pending_tickets": pending,
            "ai_configured": os.environ.get("EVIDENCEOPS_AI_ENABLED") == "1"}


@app.post("/api/search")
def search(payload: Question, x_evidenceops_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    evidence = safe_documents(payload.include_csv, include_deleted=not payload.active_only)
    if payload.active_only:
        evidence = current_evidence(evidence)
    else:
        evidence = not_deleted(evidence)
    # Optional filters: one exact source file and/or one exact version.
    chosen = []
    for row in evidence:
        if payload.source and row["source"] != payload.source:
            continue
        if payload.version and row.get("version") != payload.version:
            continue
        chosen.append(row)
    if payload.method == "bm25":
        results = search_bm25(chosen, payload.question)
    else:
        results = search_bilingual(chosen, payload.question)
    return {"method": payload.method, "results": results}


@app.get("/api/documents")
def documents(x_evidenceops_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    return {"documents": document_inventory(safe_documents(include_deleted=True))}


@app.post("/api/documents")
def add_document(payload: DocumentInput, x_evidenceops_token: str | None = Header(default=None),
                 x_operator_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    authorize_operator(x_operator_token)
    try:
        return ingest_document(STORE, payload.source, payload.version, payload.text)
    except (ValueError, KeyError) as error:
        raise HTTPException(400, str(error)) from error
    except (OSError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Document store unavailable") from error


@app.post("/api/documents/upload")
async def upload_document(request: Request, source: str, version: str,
                          x_evidenceops_token: str | None = Header(default=None),
                          x_operator_token: str | None = Header(default=None)):
    """Accept one bounded raw UTF-8 text or PDF file with approval."""
    authorize(x_evidenceops_token)
    authorize_operator(x_operator_token)
    content = await request.body()
    try:
        if source.endswith(".txt"):
            if len(content) > 100_000:
                raise ValueError("Text file exceeds 100 KB")
            return ingest_document(STORE, source, version, content.decode("utf-8-sig"), content)
        if source.endswith(".pdf"):
            return ingest_pdf(STORE, source, version, content)
        raise ValueError("Only .txt and .pdf files are supported")
    except (UnicodeError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    except (OSError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Document store unavailable") from error


@app.get("/api/documents/file")
def download_document(source: str, version: str,
                      x_evidenceops_token: str | None = Header(default=None)):
    """Download the stored original only when this version is not deleted."""
    authorize(x_evidenceops_token)
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,50}\.(txt|pdf)", source):
        raise HTTPException(400, "Invalid document filename")
    try:
        content = read_document_file(STORE, source, version)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error
    except (OSError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Document store unavailable") from error
    if source.endswith(".pdf"):
        kind = "application/pdf"
    else:
        kind = "text/plain; charset=utf-8"
    return Response(content, media_type=kind, headers={
        "Content-Disposition": f'attachment; filename="{source}"',
        "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"})


@app.get("/api/documents/history")
def local_document_history(source: str, version: str,
                           x_evidenceops_token: str | None = Header(default=None)):
    """Show stored create, supersede, and delete events for a local version."""
    authorize(x_evidenceops_token)
    try:
        return {"events": document_history(STORE, source, version)}
    except ValueError as error:
        raise HTTPException(404, str(error)) from error
    except (OSError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Document store unavailable") from error


@app.delete("/api/documents")
def remove_document(payload: DocumentRef, x_evidenceops_token: str | None = Header(default=None),
                    x_operator_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    authorize_operator(x_operator_token)
    try:
        return delete_document(STORE, payload.source, payload.version)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except (OSError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Document store unavailable") from error


@app.post("/api/documents/restore")
def restore_local_document(payload: DocumentRef, x_evidenceops_token: str | None = Header(default=None),
                           x_operator_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    authorize_operator(x_operator_token)
    try:
        return restore_document(STORE, payload.source, payload.version)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except (OSError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Document store unavailable") from error


@app.get("/api/dataset")
def dataset(x_evidenceops_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    import pyarrow.parquet as pq
    result = []
    for file in sorted((ROOT / "data" / "public").glob("*.parquet")):
        try:
            with pq.ParquetFile(file) as parquet:
                result.append(dict(source=file.name, rows=parquet.metadata.num_rows,
                                   columns=parquet.schema_arrow.names))
        except (OSError, ValueError) as error:
            raise HTTPException(503, "Invalid Parquet file") from error
    return {"files": result}


@app.get("/api/investigation")
def investigation(x_evidenceops_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    try:
        return {"report": investigation_report(DATA, current_evidence(safe_documents()))}
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(503, "Investigation unavailable") from error


@app.get("/api/evaluation")
def evaluation(x_evidenceops_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    try:
        questions = read_csv(DATA / "questions.csv", ["question_id", "question", "expected_source"])
        reviewed = read_verified_questions(REVIEWS, DATA / "questions.csv")
        reviewed_ids = []
        reviewed_sources = {}
        for row in reviewed:
            reviewed_ids.append(row["question_id"])
            reviewed_sources[row["question_id"]] = row["expected_source"]
        source_options = list(REVIEW_SOURCES)
        evidence = []
        for document in load_documents(DATA, include_csv=True):
            if document["source"] in source_options:
                evidence.append({"source": document["source"], "location": document["location"],
                                 "text": document["text"], "status": document.get("status")})
        if reviewed:
            status = "partial review"
        else:
            status = "unverified"
        return {"status": status, "questions": questions,
                "reviewed_ids": reviewed_ids, "reviewed_sources": reviewed_sources,
                "source_options": source_options, "evidence": evidence,
                "message": "Supplied answers and sources are proposals until you inspect evidence and confirm each question."}
    except (OSError, ValueError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Evaluation sources unavailable") from error


@app.post("/api/labels/review")
def save_label_review(payload: LabelReview, x_evidenceops_token: str | None = Header(default=None),
                      x_approval_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    authorize_approval(x_approval_token)
    if not payload.confirmed:
        raise HTTPException(400, "Confirm that you inspected the supporting evidence")
    try:
        return review_label(REVIEWS, DATA / "questions.csv", payload.question_id, payload.expected_source)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except (OSError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Review store unavailable") from error


@app.get("/api/experiments")
def experiments(x_evidenceops_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    try:
        questions = read_csv(DATA / "questions.csv", ["question_id", "question"])
        reviewed = read_verified_questions(REVIEWS, DATA / "questions.csv")
        return {"status": "reviewed and unreviewed source comparison",
                "results": compare_searches(safe_documents(True), questions, reviewed_questions=reviewed)}
    except (OSError, ValueError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Comparison unavailable") from error


@app.post("/api/evaluation")
def run_evaluation(verified: bool = False, x_evidenceops_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    try:
        reviewed = read_verified_questions(REVIEWS, DATA / "questions.csv")
        if not reviewed:
            raise HTTPException(409, "No saved reviews for the current questions.csv")
        report = evaluate_questions(safe_documents(True), reviewed, verified=True, search=search_bilingual)
        report["reviewed_only"] = True
        return report
    except HTTPException:
        raise
    except (OSError, ValueError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Evaluation unavailable") from error


@app.get("/api/tickets")
def tickets(x_evidenceops_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    try:
        return {"tickets": list_tickets(TICKETS)}
    except (OSError, ValueError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Ticket store unavailable") from error


@app.post("/api/tickets")
def create_ticket(payload: TicketInput, x_evidenceops_token: str | None = Header(default=None),
                  x_operator_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    authorize_operator(x_operator_token)
    try:
        report = investigation_report(DATA, current_evidence(safe_documents()))
    except (OSError, ValueError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Investigation unavailable") from error
    try:
        return prepare_ticket(TICKETS, payload.question, report, payload.request_key)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except (OSError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Ticket store unavailable") from error


@app.post("/api/tickets/{ticket_id}/approve")
def approve(ticket_id: int, x_evidenceops_token: str | None = Header(default=None),
            x_approval_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    authorize_approval(x_approval_token)
    try:
        return approve_ticket(TICKETS, ticket_id)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except (OSError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Ticket store unavailable") from error


@app.post("/api/tickets/{ticket_id}/{decision}")
def decide(ticket_id: int, decision: str, x_evidenceops_token: str | None = Header(default=None),
           x_approval_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    authorize_approval(x_approval_token)
    try:
        return change_ticket(TICKETS, ticket_id, decision)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except (OSError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Ticket store unavailable") from error


@app.get("/api/tickets/{ticket_id}/history")
def history(ticket_id: int, x_evidenceops_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    try:
        return {"events": ticket_history(TICKETS, ticket_id)}
    except ValueError as error:
        raise HTTPException(404, str(error)) from error
    except (OSError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Ticket store unavailable") from error


def ai_error(error):
    """Turn a real configuration, Ollama or embedding failure into one API error."""
    return HTTPException(503, f"AI unavailable: {error}")


@app.get("/api/ai/status")
def ai_state(x_evidenceops_token: str | None = Header(default=None)):
    """Report the configured, installed models and the stored vector index."""
    authorize(x_evidenceops_token)
    try:
        return ai_status()
    except (RuntimeError, ValueError, OSError, sqlite3.DatabaseError) as error:
        raise ai_error(error) from error


@app.post("/api/ai/index")
def ai_index(rebuild: bool = False, x_evidenceops_token: str | None = Header(default=None),
             x_operator_token: str | None = Header(default=None)):
    """Embed any chunk with no cached vector; rebuild drops the stored vectors first."""
    authorize(x_evidenceops_token)
    authorize_operator(x_operator_token)
    try:
        index = build_index(evidence_documents(), ai_settings(), rebuild)
    except (RuntimeError, ValueError, OSError, sqlite3.DatabaseError) as error:
        raise ai_error(error) from error
    return {"model_key": index["model_key"], "dimensions": index["dimensions"],
            "chunks": index["chunks"], "embedded": index["embedded"], "reused": index["reused"]}


def chosen_documents(dataset):
    """The evidence for one mode. The two modes are never searched together."""
    if dataset == "steel":
        return steel_documents()
    return evidence_documents()


@app.post("/api/ask")
def ask(payload: AskInput, x_evidenceops_token: str | None = Header(default=None)):
    """Retrieve real evidence, generate a grounded answer, and verify its citations."""
    authorize(x_evidenceops_token)
    try:
        return answer_question(payload.question, payload.method, chosen_documents(payload.dataset))
    except (RuntimeError, ValueError, OSError, sqlite3.DatabaseError) as error:
        raise ai_error(error) from error


@app.post("/api/retrieval/compare")
def retrieval_compare(payload: AskInput, x_evidenceops_token: str | None = Header(default=None)):
    """Rank the same chunks with all four methods for one question."""
    authorize(x_evidenceops_token)
    try:
        return compare_methods(payload.question, chosen_documents(payload.dataset))
    except (RuntimeError, ValueError, OSError, sqlite3.DatabaseError) as error:
        raise ai_error(error) from error


@app.get("/api/steel")
def steel_facts(x_evidenceops_token: str | None = Header(default=None)):
    """Mode B facts, calculated in Python from the real UCI files. No model is used."""
    authorize(x_evidenceops_token)
    try:
        return summarize(load_dataset(STEEL_DIR))
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(503, f"Steel dataset unavailable: {error}") from error


@app.post("/api/steel/download")
def steel_download(x_evidenceops_token: str | None = Header(default=None),
                   x_operator_token: str | None = Header(default=None)):
    """Download the official UCI files. The address and file hashes are fixed in steel_faults.py."""
    authorize(x_evidenceops_token)
    authorize_operator(x_operator_token)
    try:
        return {"saved": download_dataset(STEEL_DIR)}
    except (OSError, ValueError) as error:
        raise HTTPException(503, f"Download failed: {error}") from error


@app.post("/api/agent")
def agent(payload: AgentInput, x_evidenceops_token: str | None = Header(default=None),
          x_operator_token: str | None = Header(default=None)):
    """Run the bounded agent; a draft needs the operator token, approval still needs a human."""
    authorize(x_evidenceops_token)
    allow_draft = bool(x_operator_token)
    if allow_draft:
        authorize_operator(x_operator_token)
    try:
        return run_agent(payload.request, allow_draft)
    except (RuntimeError, ValueError, OSError, sqlite3.DatabaseError) as error:
        raise ai_error(error) from error


@app.get("/api/ai/runs")
def ai_runs(x_evidenceops_token: str | None = Header(default=None)):
    """Read the latest saved AI runs, including withheld and failed ones."""
    authorize(x_evidenceops_token)
    try:
        return {"runs": list_ai_runs(RUNS)}
    except (OSError, ValueError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "AI run history unavailable") from error


@app.post("/api/workflow")
def workflow(payload: WorkflowInput, x_evidenceops_token: str | None = Header(default=None),
             x_operator_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    if payload.kind == "sop_conflict":
        authorize_operator(x_operator_token)
    try:
        result = run_workflow(payload.question, payload.kind, TICKETS, STORE)
        record_workflow_run(TICKETS, result)
        return result
    except (OSError, ValueError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Workflow unavailable") from error


@app.get("/api/workflows/history")
def workflow_history(x_evidenceops_token: str | None = Header(default=None)):
    authorize(x_evidenceops_token)
    try:
        return {"runs": list_workflow_runs(TICKETS)}
    except (OSError, ValueError, sqlite3.DatabaseError) as error:
        raise HTTPException(503, "Workflow history unavailable") from error
