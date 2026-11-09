"""Small deterministic investigation workflows. No LLM is called."""
import json
import re
import sqlite3
from contextlib import closing
from datetime import date
from math import isfinite
from data_io import DATA, LOCAL, load_documents, read_csv
from retrieval import search_bilingual, current_evidence
from investigation import investigation_report, prepare_ticket


def find_evidence(state):
    """Fetch up to three keyword matches and record the action."""
    documents = current_evidence(load_documents(DATA, True, state["documents_file"]))
    return {"evidence": search_bilingual(documents, state["question"]),
            "trace": ["search_documents"]}


def compare_deployments():
    """Compare actual Line B v3/v4 deployment settings with record citations."""
    rows = read_csv(DATA / "deployments.csv", ["deployment_id", "date", "line", "model",
                    "defect_score_threshold", "image_size", "status"])
    versions = {}
    seen_ids = set()
    lines = ["Line B deployment comparison (deterministic; no LLM)"]
    for number, row in enumerate(rows, 1):
        date.fromisoformat(row["date"])
        if row["deployment_id"] in seen_ids:
            raise ValueError(f"Duplicate deployment ID in record {number}")
        seen_ids.add(row["deployment_id"])
        threshold = float(row["defect_score_threshold"])
        size = int(row["image_size"])
        if not isfinite(threshold) or not 0 <= threshold <= 1 or size <= 0:
            raise ValueError(f"Invalid deployment settings in record {number}")
        if row["line"] == "B" and row["model"] in ("v3", "v4"):
            if row["model"] in versions:
                raise ValueError(f"Ambiguous Line B {row['model']} deployment")
            versions[row["model"]] = (number, row, threshold, size)
            lines.append(f"{row['model']}: {row['date']}, threshold {threshold}, image size {size}, "
                         f"status {row['status']} [deployments.csv record {number}]")
    missing = []
    for model in ("v3", "v4"):
        if model not in versions:
            missing.append(f"Line B {model} deployment record")
    if not missing:
        before = versions["v3"]
        after = versions["v4"]
        change = after[2] - before[2]
        lines.append(f"Threshold change: {change:+.2f}; image size {before[3]} -> {after[3]} "
                     f"[deployments.csv records {before[0]}, {after[0]}]")
    missing.append("Named approver and proof of the live production setting")
    lines.append("Missing: " + "; ".join(missing))
    return "\n".join(lines), missing


def compare_sop_versions(local_file):
    """Find actual SOP threshold differences without calling history a live conflict."""
    documents = load_documents(DATA, local_file=local_file, include_deleted=True)
    lines = ["Line B SOP version comparison (deterministic; no LLM)"]
    active = set()
    historical = set()
    for document in documents:
        if not document["source"].startswith("sop") or document["status"] == "DELETED":
            continue
        match = re.search(r"(?:threshold(?: was)?|閾值為)\s*(0\.\d+)", document["text"], re.IGNORECASE)
        if not match:
            continue
        value = float(match.group(1))
        status = document["status"]
        lines.append(f"{document['source']} {document['version']}: threshold {value}, status {status} "
                     f"[{document['source']} {document['location']}]")
        if status.upper() in ("ACTIVE", "CURRENT", "有效"):
            active.add(value)
        else:
            historical.add(value)
    missing = []
    if not active:
        missing.append("An active SOP with an extractable threshold")
    elif len(active) > 1:
        lines.append("Conflict: active SOPs specify different thresholds.")
    else:
        lines.append("No differing threshold was found among active SOPs.")
    if historical and active and historical != active:
        lines.append("Historical and active thresholds differ; this alone is not a simultaneous active conflict.")
    missing.append("Proof of the live production setting and named change approval")
    lines.append("Missing: " + "; ".join(missing))
    return "\n".join(lines), missing


def calculate_report(state):
    """Calculate a source-linked report without editing ticket state."""
    kind = state.get("kind", "inspection")
    if kind == "inspection":
        documents = current_evidence(load_documents(DATA, local_file=state["documents_file"]))
        report = investigation_report(DATA, documents)
        return {"report": report, "missing": ["Named deployment approver and live setting verification"],
                "trace": state["trace"] + ["calculate_report", "stop_before_ticket_approval"]}
    if kind == "deployment":
        report, missing = compare_deployments()
        return {"report": report, "missing": missing,
                "trace": state["trace"] + ["compare_deployments", "stop_before_ticket_approval"]}
    report, missing = compare_sop_versions(state["documents_file"])
    ticket = prepare_ticket(state["ticket_path"], state["question"], report)
    return {"report": report, "missing": missing, "ticket": ticket,
            "trace": state["trace"] + ["compare_sop_versions", "prepare_ticket", "stop_before_ticket_approval"]}


def run_workflow(question, kind="inspection", ticket_path=None, documents_file=None):
    """Return public findings and trace, without internal local file paths."""
    if not question.strip() or len(question) > 1000:
        raise ValueError("Question must contain 1 to 1000 characters")
    if kind not in ("inspection", "deployment", "sop_conflict"):
        raise ValueError("Choose inspection, deployment, or sop_conflict")
    if ticket_path is None:
        ticket_path = LOCAL / "tickets.db"
    if documents_file is None:
        documents_file = LOCAL / "documents.db"
    state = {"question": question, "kind": kind, "ticket_path": ticket_path,
             "documents_file": documents_file}
    state.update(find_evidence(state))
    state.update(calculate_report(state))
    result = {"question": question, "kind": kind, "evidence": state["evidence"],
              "report": state["report"], "missing": state["missing"], "trace": state["trace"]}
    if "ticket" in state:
        result["ticket"] = state["ticket"]
    return result


def record_workflow_run(path, result):
    """Save one completed deterministic run in the local ticket database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=3)) as connection:
        with connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS workflow_runs
                (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, question TEXT NOT NULL,
                 report TEXT NOT NULL, evidence_json TEXT NOT NULL,
                 missing_json TEXT NOT NULL, trace_json TEXT NOT NULL,
                 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
            cursor = connection.execute("""INSERT INTO workflow_runs
                (kind, question, report, evidence_json, missing_json, trace_json)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (result["kind"], result["question"], result["report"],
                 json.dumps(result["evidence"], ensure_ascii=False),
                 json.dumps(result["missing"], ensure_ascii=False),
                 json.dumps(result["trace"], ensure_ascii=False)))
    return cursor.lastrowid


def list_workflow_runs(path):
    """Read the latest 50 saved reports without changing ticket states."""
    if not path.exists():
        return []
    with closing(sqlite3.connect(path, timeout=3)) as connection:
        table = connection.execute("""SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='workflow_runs'""").fetchone()
        if not table:
            return []
        rows = connection.execute("""SELECT id, kind, question, report, evidence_json,
            missing_json, trace_json, created_at FROM workflow_runs
            ORDER BY id DESC LIMIT 50""").fetchall()
    history = []
    for row in rows:
        history.append({"id": row[0], "kind": row[1], "question": row[2], "report": row[3],
                        "evidence": json.loads(row[4]), "missing": json.loads(row[5]),
                        "trace": json.loads(row[6]), "at": row[7]})
    return history
