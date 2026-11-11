"""Deterministic calculations over the supplied synthetic records."""
from datetime import date
from math import isfinite
import sqlite3
import hashlib
from contextlib import closing
from data_io import read_csv, table_columns


def summarize_metrics(rows):
    """Validate counts and return totals grouped by line and model."""
    groups = {}
    seen = set()
    for number, row in enumerate(rows, 1):
        date.fromisoformat(row["date"])
        if not row["line"].strip() or not row["camera"].strip() or not row["model"].strip():
            raise ValueError(f"Missing identity in metrics record {number}")
        identity = (row["date"], row["line"], row["camera"], row["model"])
        if identity in seen:
            raise ValueError(f"Duplicate metrics record {number}")
        seen.add(identity)
        inspected = int(row["inspected"])
        rejected = int(row["rejected"])
        audited = int(row["verified_good_rejected"])
        if not 0 <= audited <= rejected <= inspected:
            raise ValueError(f"Invalid counts in metrics record {number}")
        key = (row["line"], row["model"])
        if key not in groups:
            groups[key] = dict(inspected=0, rejected=0, records=[])
        groups[key]["inspected"] += inspected
        groups[key]["rejected"] += rejected
        groups[key]["records"].append(number)
    for group in groups.values():
        group["rate"] = None
        if group["inspected"]:
            group["rate"] = 100 * group["rejected"] / group["inspected"]
    return groups


def investigation_report(folder, documents):
    """Return a source-linked text report; invalid inputs abort calculations."""
    metrics = read_csv(folder / "metrics.csv", ["date", "line", "camera", "model",
                       "inspected", "rejected", "verified_good_rejected"])
    deployments = read_csv(folder / "deployments.csv", ["deployment_id", "date", "line",
                           "model", "defect_score_threshold", "image_size", "status"])
    groups = summarize_metrics(metrics)
    deployment_ids = set()
    line_b_versions = {}
    output = ["Synthetic evidence investigation (deterministic Python; no LLM)"]
    for key in sorted(groups):
        line, model = key
        group = groups[key]
        rate = "undefined (zero inspected)"
        if group["rate"] is not None:
            rate = f"{group['rate']:.2f}%"
        output.append(f"Line {line}, {model}: {group['rejected']}/{group['inspected']} = {rate} "
                      f"[metrics.csv records {group['records']}]")
    for number, row in enumerate(deployments, 1):
        date.fromisoformat(row["date"])
        if row["deployment_id"] in deployment_ids:
            raise ValueError(f"Duplicate deployment ID in record {number}")
        deployment_ids.add(row["deployment_id"])
        threshold = float(row["defect_score_threshold"])
        if not isfinite(threshold) or not 0 <= threshold <= 1 or int(row["image_size"]) <= 0:
            raise ValueError(f"Invalid deployment settings at record {number}")
        output.append(f"{row['deployment_id']}: {row['date']}, Line {row['line']}, "
                      f"{row['model']}, threshold {threshold}, size {row['image_size']}, "
                      f"status {row['status']} [deployments.csv record {number}]")
        if row["line"] == "B" and row["model"] in ("v3", "v4"):
            if row["model"] in line_b_versions:
                raise ValueError(f"Multiple Line B {row['model']} deployments; comparison is ambiguous")
            line_b_versions[row["model"]] = {"record": number, "threshold": threshold,
                                              "image_size": int(row["image_size"])}
    if "v3" in line_b_versions and "v4" in line_b_versions:
        old = line_b_versions["v3"]
        new = line_b_versions["v4"]
        output.append(f"Line B deployment settings: threshold {old['threshold']} -> {new['threshold']}, "
                      f"image size {old['image_size']} -> {new['image_size']} "
                      f"[deployments.csv records {old['record']}, {new['record']}]")
    before = groups.get(("B", "v3"))
    after = groups.get(("B", "v4"))
    both_found = before is not None and after is not None
    if both_found and before["rate"] is not None and after["rate"] is not None:
        difference = after["rate"] - before["rate"]
        output.append(f"Line B v4 minus v3: {difference:+.2f} percentage points "
                      f"[metrics.csv records {before['records'] + after['records']}]")
    for document in documents:
        if document["source"].startswith("sop"):
            output.append(f"[{document['source']} {document['location']}]\n{document['text']}")
    output.append("Observed association does not establish cause. Audited rejected-good counts "
                  "do not supply the denominator for a true false-rejection rate. "
                  "No named deployment approver field is supplied. "
                  "Deployment and SOP effective dates must be interpreted separately.")
    return "\n".join(output)


def setup_tickets(connection):
    """Create tables and migrate the old prepared state without losing tickets."""
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""CREATE TABLE IF NOT EXISTS tickets
            (id INTEGER PRIMARY KEY, question TEXT NOT NULL, report TEXT NOT NULL,
             state TEXT NOT NULL, request_key TEXT UNIQUE)""")
        columns = table_columns(connection, "tickets")
        if "request_key" not in columns:
            connection.execute("ALTER TABLE tickets ADD COLUMN request_key TEXT")
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS tickets_request_key ON tickets(request_key)")
        connection.execute("""CREATE TABLE IF NOT EXISTS ticket_events
            (id INTEGER PRIMARY KEY, ticket_id INTEGER NOT NULL REFERENCES tickets(id),
             old_state TEXT, new_state TEXT NOT NULL, actor TEXT NOT NULL,
             event_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        # Older versions called a waiting ticket "prepared"; rename it to "pending" once.
        old_ids = []
        for row in connection.execute("SELECT id FROM tickets WHERE state='prepared'"):
            old_ids.append(row[0])
        for ticket_id in old_ids:
            connection.execute("UPDATE tickets SET state='pending' WHERE id=?", (ticket_id,))
            connection.execute("""INSERT INTO ticket_events (ticket_id, old_state, new_state, actor)
                VALUES (?, 'prepared', 'pending', 'migration')""", (ticket_id,))


def prepare_ticket(path, question, report, request_key=None):
    """Save a pending draft once for a question/report pair or explicit key."""
    if not question.strip() or len(question) > 1000 or not report.strip():
        raise ValueError("A short question and a report are required")
    key = request_key or hashlib.sha256((question.strip() + "\n" + report).encode("utf-8")).hexdigest()
    if not 1 <= len(key) <= 100:
        raise ValueError("Invalid request key")
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=3)) as connection:
        setup_tickets(connection)
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT id, state, question, report FROM tickets WHERE request_key=?", (key,)).fetchone()
            if previous:
                if previous[2] != question.strip() or previous[3] != report:
                    raise ValueError("Request key already belongs to a different ticket")
                return dict(id=previous[0], state=previous[1], duplicate=True)
            cursor = connection.execute("INSERT INTO tickets (question, report, state, request_key) VALUES (?, ?, 'pending', ?)",
                                        (question.strip(), report, key))
            connection.execute("""INSERT INTO ticket_events (ticket_id, old_state, new_state, actor)
                VALUES (?, NULL, 'pending', 'operator')""", (cursor.lastrowid,))
            return dict(id=cursor.lastrowid, state="pending", duplicate=False)


def list_tickets(path):
    """Read saved ticket summaries."""
    if not path.exists():
        return []
    with closing(sqlite3.connect(path, timeout=3)) as connection:
        setup_tickets(connection)
        rows = connection.execute("SELECT id, question, state FROM tickets ORDER BY id DESC").fetchall()
    tickets = []
    for row in rows:
        tickets.append({"id": row[0], "question": row[1], "state": row[2]})
    return tickets


def approve_ticket(path, ticket_id):
    """Approve one pending ticket after the API checks the approval token."""
    return change_ticket(path, ticket_id, "approved")


def change_ticket(path, ticket_id, decision):
    """Move pending to approved/rejected, or approved to completed, once."""
    old_state = {"approved": "pending", "rejected": "pending", "completed": "approved"}.get(decision)
    if old_state is None:
        raise ValueError("Invalid ticket decision")
    if not path.exists():
        raise ValueError("Ticket not found")
    with closing(sqlite3.connect(path, timeout=3)) as connection:
        setup_tickets(connection)
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute("UPDATE tickets SET state=? WHERE id=? AND state=?",
                                        (decision, ticket_id, old_state))
            if cursor.rowcount != 1:
                raise ValueError("Ticket not found or transition already completed")
            connection.execute("""INSERT INTO ticket_events (ticket_id, old_state, new_state, actor)
                VALUES (?, ?, ?, 'approver')""", (ticket_id, old_state, decision))
    return dict(id=ticket_id, state=decision)


def ticket_history(path, ticket_id):
    """Read the persisted decision history for one ticket."""
    if not path.exists():
        raise ValueError("Ticket not found")
    with closing(sqlite3.connect(path, timeout=3)) as connection:
        setup_tickets(connection)
        rows = connection.execute("""SELECT old_state, new_state, actor, event_at
            FROM ticket_events WHERE ticket_id=? ORDER BY id""", (ticket_id,)).fetchall()
        exists = connection.execute("SELECT 1 FROM tickets WHERE id=?", (ticket_id,)).fetchone()
        if not exists:
            raise ValueError("Ticket not found")
    events = []
    for row in rows:
        events.append({"old_state": row[0], "new_state": row[1], "actor": row[2], "at": row[3]})
    return events
