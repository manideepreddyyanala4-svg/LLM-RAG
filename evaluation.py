"""Evaluate only labels explicitly confirmed by the user."""
import json
import hashlib
import platform
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from retrieval import search_documents
from retrieval import search_bilingual, search_bm25
from data_io import read_csv

REVIEW_SOURCES = ("sop_v1.txt", "sop_v2_en.txt", "sop_v2_zh.txt", "metrics.csv", "deployments.csv")


def evaluate_questions(documents, questions, verified=False, search=search_documents):
    """Return counts and per-question results, or refuse unverified labels."""
    if not verified:
        raise ValueError("Human verification is not recorded. Review labels before confirming.")
    report = dict(total=0, top1=0, top3=0, skipped=0, details=[])
    for row in questions:
        expected = set()
        for source in row["expected_source"].split(";"):
            if source.strip():
                expected.add(source.strip())
        if not expected or not row["question"].strip():
            report["skipped"] += 1
            continue
        results = search(documents, row["question"])
        sources = [result["source"] for result in results]
        first = bool(sources and sources[0] in expected)
        any_hit = bool(expected & set(sources))
        report["total"] += 1
        report["top1"] += int(first)
        report["top3"] += int(any_hit)
        report["details"].append(dict(id=row["question_id"], question=row["question"],
                                      expected=sorted(expected), results=results,
                                      top1=first, top3=any_hit))
    if report["total"]:
        report["top1_percent"] = round(100 * report["top1"] / report["total"], 2)
        report["top3_percent"] = round(100 * report["top3"] / report["total"], 2)
    else:
        report["top1_percent"] = 0.0
        report["top3_percent"] = 0.0
    return report


def print_evaluation(report):
    """Print counts, percentages and failures from an evaluation dictionary."""
    total = report["total"]
    print(f"Total labeled questions tested: {total}; skipped: {report['skipped']}")
    for key in ("top1", "top3"):
        percent = 100 * report[key] / total if total else 0
        print(f"{key}: {report[key]}/{total} ({percent:.1f}%)")
    print("Top-1 source accuracy; top-3 any-source hit rate (not full multi-source recall).")
    print("These measure source retrieval, not answer correctness or paragraph sufficiency.")
    for row in report["details"]:
        if not row["top1"] or not row["top3"]:
            print(f"{row['id']}: {row['question']} | top1={row['top1']}, top3={row['top3']}")
            print("Expected:", row["expected"])
            print("Retrieved:", row["results"])


def compare_searches(documents, questions, log_file=None, reviewed_questions=None):
    """Record ranks and time; mark misses only for saved reviews."""
    rows = []
    methods = {"keyword": search_bilingual, "bm25": search_bm25}
    reviewed = {}
    for question in reviewed_questions or []:
        sources = set()
        for source in question["expected_source"].split(";"):
            if source.strip():
                sources.add(source.strip())
        reviewed[question["question_id"]] = sources
    for question in questions:
        for name, search in methods.items():
            start = time.perf_counter()
            results = search(documents, question["question"])
            elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
            matches = []
            for rank, result in enumerate(results, 1):
                matches.append({"rank": rank, "source": result["source"], "location": result["location"],
                                "version": result.get("version"), "score": result["score"]})
            expected = reviewed.get(question["question_id"])
            top1 = None
            top3 = None
            failure = None
            if expected is not None:
                sources = [match["source"] for match in matches]
                top1 = bool(sources and sources[0] in expected)
                top3 = bool(expected & set(sources))
                if not top3:
                    failure = "reviewed source absent from top three"
                elif not top1:
                    failure = "reviewed source absent from rank one"
            elif not matches:
                failure = "no lexical matches; correctness unscored"
            rows.append({"id": question["question_id"], "question": question["question"],
                         "method": name, "results": matches, "elapsed_ms": elapsed_ms,
                         "reviewed": expected is not None, "top1": top1, "top3": top3,
                         "failure": failure})
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        evidence = []
        for document in documents:
            evidence.append([document["source"], document["location"], document["text"],
                             document.get("version"), document.get("status"), document.get("hash")])
        questions_used = []
        for question in questions:
            questions_used.append([question["question_id"], question["question"]])
        evidence_hash = hashlib.sha256(json.dumps(evidence, ensure_ascii=False).encode("utf-8")).hexdigest()
        questions_hash = hashlib.sha256(json.dumps(questions_used, ensure_ascii=False).encode("utf-8")).hexdigest()
        record = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                  "python": platform.python_version(), "backend": "local lexical search",
                  "model": None, "evidence_sha256": evidence_hash,
                  "questions_sha256": questions_hash, "results": rows}
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return rows


def review_label(store, csv_file, question_id, expected_source):
    """Save a manual label with hashes of the question CSV and cited files."""
    questions = read_csv(csv_file, ["question_id", "question", "expected_source"])
    selected = None
    for question in questions:
        if question["question_id"] == question_id:
            selected = question
            break
    if selected is None:
        raise ValueError("Unknown question ID")
    sources = [source.strip() for source in expected_source.split(";")]
    if not sources or any(source not in REVIEW_SOURCES for source in sources):
        raise ValueError("Choose one or more approved source filenames")
    source_hashes = {}
    for source in sources:
        if not (csv_file.parent / source).is_file():
            raise ValueError(f"Source file missing: {source}")
        source_hashes[source] = hashlib.sha256((csv_file.parent / source).read_bytes()).hexdigest()
    digest = hashlib.sha256(csv_file.read_bytes()).hexdigest()
    store.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(store, timeout=3)) as connection:
        with connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS reviewed_labels
                (question_id TEXT PRIMARY KEY, question TEXT NOT NULL,
                 expected_source TEXT NOT NULL, csv_hash TEXT NOT NULL,
                 source_hashes TEXT, reviewed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
            columns = [row[1] for row in connection.execute("PRAGMA table_info(reviewed_labels)")]
            if "source_hashes" not in columns:
                connection.execute("ALTER TABLE reviewed_labels ADD COLUMN source_hashes TEXT")
            connection.execute("""INSERT INTO reviewed_labels
                (question_id, question, expected_source, csv_hash, source_hashes)
                VALUES (?, ?, ?, ?, ?) ON CONFLICT(question_id) DO UPDATE SET
                question=excluded.question, expected_source=excluded.expected_source,
                csv_hash=excluded.csv_hash, source_hashes=excluded.source_hashes,
                reviewed_at=CURRENT_TIMESTAMP""",
                (question_id, selected["question"], "; ".join(sources), digest,
                 json.dumps(source_hashes, sort_keys=True)))
    return {"id": question_id, "sources": sources, "status": "reviewed"}


def read_verified_questions(store, csv_file):
    """Return reviews only while both questions and cited files are unchanged."""
    if not store.exists():
        return []
    digest = hashlib.sha256(csv_file.read_bytes()).hexdigest()
    questions = read_csv(csv_file, ["question_id", "question", "expected_source"])
    with closing(sqlite3.connect(store, timeout=3)) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(reviewed_labels)")]
        if "source_hashes" not in columns:
            return []  # Earlier reviews need a fresh evidence inspection.
        rows = connection.execute("""SELECT question_id, question, expected_source, csv_hash, source_hashes
            FROM reviewed_labels""").fetchall()
    reviewed = {}
    for question_id, question, sources, saved_hash, saved_sources in rows:
        if saved_hash != digest or not saved_sources:
            continue
        try:
            source_hashes = json.loads(saved_sources)
        except ValueError:
            continue
        if not isinstance(source_hashes, dict):
            continue
        current_sources = {source.strip() for source in sources.split(";") if source.strip()}
        if not current_sources or not current_sources.issubset(set(REVIEW_SOURCES)):
            continue
        if current_sources != set(source_hashes):
            continue
        unchanged = True
        for source in current_sources:
            path = csv_file.parent / source
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[source]:
                unchanged = False
                break
        if unchanged:
            reviewed[question_id] = {"question_id": question_id, "question": question,
                                     "expected_source": sources}
    results = []
    for question in questions:
        if question["question_id"] in reviewed:
            results.append(reviewed[question["question_id"]])
    return results
