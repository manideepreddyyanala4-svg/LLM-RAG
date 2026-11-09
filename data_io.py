"""Read local evidence without changing the source files."""
import csv
import hashlib
from io import BytesIO
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from contextlib import closing
from pathlib import Path
from pprint import pprint
from pypdf import PdfReader
from pypdf.errors import PyPdfError

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("EVIDENCEOPS_DATA_DIR", ROOT / "EvidenceOps_Data_Starter" / "data"))
if not DATA.is_absolute():
    raise ValueError("EVIDENCEOPS_DATA_DIR must be an absolute path")
configured_local = os.environ.get("EVIDENCEOPS_LOCAL_DIR")
LOCAL = Path(configured_local) if configured_local else ROOT / "data" / "local"
if not LOCAL.is_absolute():
    raise ValueError("EVIDENCEOPS_LOCAL_DIR must be an absolute path")


def read_csv(path, required):
    """Return CSV dictionaries; reject missing columns and malformed rows."""
    if path.stat().st_size > 5_000_000:
        raise ValueError(f"{path.name}: this loader is for small CSV files only")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name}: missing required columns {required}")
        if reader.fieldnames and len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path.name}: duplicate column names")
        rows = []
        question_ids = set()
        for number, row in enumerate(reader, 1):
            if None in row or None in row.values():
                raise ValueError(f"{path.name}: malformed record {number}")
            for key in required:
                if not row[key].strip():
                    raise ValueError(f"{path.name}: empty {key} in record {number}")
            if "question_id" in required:
                if row["question_id"] in question_ids:
                    raise ValueError(f"{path.name}: duplicate question ID in record {number}")
                question_ids.add(row["question_id"])
            rows.append(row)
    return rows


def load_documents(folder=None, include_csv=False, local_file=None, include_deleted=False):
    """Return source, location and text dictionaries for SOPs and optional CSVs."""
    folder = DATA if folder is None else folder
    documents = []
    if not folder.is_dir():
        raise FileNotFoundError(f"Evidence folder missing: {folder}")
    for path in sorted(folder.glob("sop*.txt")):
        content = path.read_bytes()
        text = content.decode("utf-8-sig")
        header = text.split("\n\n", 1)[0]
        version = re.search(r"(?:DOCUMENT ID|文件編號)\s*[:：]\s*([^\n]+)", header)
        status = re.search(r"(?:STATUS|狀態)\s*[:：]\s*([^\n]+)", header)
        digest = hashlib.sha256(content).hexdigest()
        for number, paragraph in enumerate(re.split(r"\n\s*\n", text), 1):
            if paragraph.strip():
                documents.append(dict(source=path.name, location=f"paragraph {number}",
                                      text=paragraph.strip(), version=version.group(1).strip() if version else "unknown",
                                      status=status.group(1).strip() if status else "unknown", hash=digest))
    if local_file is not None:
        documents.extend(read_local_documents(local_file, include_deleted))
    if include_csv:
        # Questions and expected answers must never become searchable evidence.
        for name in ("metrics.csv", "deployments.csv"):
            digest = hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for number, row in enumerate(read_csv(folder / name, ["date", "line", "model"]), 1):
                fields = []
                for key, value in row.items():
                    fields.append(f"{key}: {value}")
                documents.append(dict(source=name, location=f"record {number}",
                                      text="; ".join(fields), hash=digest, version=digest[:12],
                                      line=row["line"], record_model=row["model"]))
    return documents


def ingest_document(store, source, version, text, original=None):
    """Insert text and retain the exact uploaded bytes when provided."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,50}\.txt", source):
        raise ValueError("Use a simple .txt filename")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Version and nonempty text (at most 100 KB) are required")
    content = text.encode("utf-8") if original is None else original
    if not isinstance(content, bytes) or len(content) > 100_000:
        raise ValueError("Text file exceeds 100 KB")
    if content.decode("utf-8-sig") != text:
        raise ValueError("Original text bytes do not match the supplied text")
    chunks = []
    for number, paragraph in enumerate(re.split(r"\n\s*\n", text), 1):
        if paragraph.strip():
            chunks.append((f"paragraph {number}", paragraph.strip()))
    return save_document(store, source, version, content, chunks)


def ingest_pdf(store, source, version, content):
    """Extract PDF text, using optional OCR for pages with no text layer."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,50}\.pdf", source):
        raise ValueError("Use a simple .pdf filename")
    if not isinstance(content, bytes) or not content.startswith(b"%PDF-") or len(content) > 2_000_000:
        raise ValueError("A PDF of at most 2 MB is required")
    try:
        reader = PdfReader(BytesIO(content), strict=True)
        if reader.is_encrypted or not 1 <= len(reader.pages) <= 20:
            raise ValueError("PDF must have 1 to 20 readable pages and no encryption")
        chunks = []
        scanned_pages = []
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                chunks.append((f"page {number}", text.strip()))
            else:
                scanned_pages.append(number)
        if scanned_pages:
            recognized = ocr_pdf_pages(content, scanned_pages, store)
            for number in scanned_pages:
                chunks.append((f"page {number}", recognized[number]))
        chunks.sort(key=lambda row: int(row[0].split()[1]))
        if sum(len(text) for _, text in chunks) > 100_000:
            raise ValueError("Extracted PDF text exceeds 100 KB")
    except (OSError, KeyError, TypeError, PyPdfError) as error:
        raise ValueError(f"Cannot read PDF: {error}") from error
    return save_document(store, source, version, content, chunks)


def ocr_pdf_pages(content, page_numbers, store):
    """Use optional Poppler and Tesseract executables for image-only pages."""
    renderer = shutil.which("pdftoppm")
    recognizer = shutil.which("tesseract")
    if not renderer or not recognizer:
        raise ValueError("PDF page has no extractable text; OCR needs pdftoppm and Tesseract")
    language = os.environ.get("EVIDENCEOPS_OCR_LANG", "eng")
    if not re.fullmatch(r"[A-Za-z0-9_+]{1,40}", language):
        raise ValueError("Invalid EVIDENCEOPS_OCR_LANG")
    store.parent.mkdir(parents=True, exist_ok=True)
    recognized = {}
    deadline = time.monotonic() + 60
    with tempfile.TemporaryDirectory(dir=store.parent) as folder:
        pdf = Path(folder) / "input.pdf"
        pdf.write_bytes(content)
        for number in page_numbers:
            image = Path(folder) / f"page-{number}.png"
            base = image.with_suffix("")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("PDF OCR exceeded 60 seconds")
            try:
                rendered = subprocess.run([renderer, "-f", str(number), "-l", str(number),
                                           "-singlefile", "-scale-to", "1600", "-png", str(pdf), str(base)],
                                          capture_output=True, timeout=min(10, remaining))
                if rendered.returncode != 0 or not image.is_file():
                    raise ValueError(f"PDF rendering failed on page {number}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("PDF OCR exceeded 60 seconds")
                result = subprocess.run([recognizer, str(image), "stdout", "-l", language],
                                        capture_output=True, timeout=min(10, remaining))
            except (OSError, subprocess.TimeoutExpired) as error:
                raise ValueError(f"PDF OCR unavailable or timed out on page {number}") from error
            if result.returncode != 0:
                raise ValueError(f"PDF OCR failed on page {number}; check installed language data")
            text = result.stdout.decode("utf-8", errors="replace").strip()
            if not text:
                raise ValueError(f"PDF OCR found no readable text on page {number}")
            recognized[number] = text
    return recognized


def save_document(store, source, version, content, chunks):
    """Save one immutable file version and its cited chunks atomically."""
    if (DATA / source).exists():
        raise ValueError("The original starter document is read-only")
    if not re.fullmatch(r"[a-zA-Z0-9._-]{1,40}", version):
        raise ValueError("Use a simple version name of at most 40 characters")
    digest = hashlib.sha256(content).hexdigest()
    store.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(store, timeout=3)) as connection:
        setup_documents(connection, store)
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT digest, status FROM local_documents WHERE source=? AND version=?",
                                          (source, version)).fetchone()
            if previous:
                if previous[0] == digest and previous[1] == "CURRENT":
                    return {"status": "duplicate", "hash": digest}
                if previous[0] == digest:
                    raise ValueError("That version is inactive; use a new version name")
                raise ValueError("That source/version already exists with different content")
            old_ids = [row[0] for row in connection.execute(
                "SELECT id FROM local_documents WHERE source=? AND status='CURRENT'", (source,))]
            connection.execute("UPDATE local_documents SET status='OLD' WHERE source=? AND status='CURRENT'", (source,))
            for document_id in old_ids:
                record_document_event(connection, document_id, "superseded")
            cursor = connection.execute("""INSERT INTO local_documents (source, version, digest, status, content)
                VALUES (?, ?, ?, 'CURRENT', ?)""", (source, version, digest, content))
            record_document_event(connection, cursor.lastrowid, "created")
            for location, text in chunks:
                connection.execute("INSERT INTO local_chunks (document_id, location, text) VALUES (?, ?, ?)",
                                   (cursor.lastrowid, location, text))
    return {"status": "added", "hash": digest}


def setup_documents(connection, store):
    """Create local tables and import a prior JSON store once, if present."""
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""CREATE TABLE IF NOT EXISTS local_documents
            (id INTEGER PRIMARY KEY, source TEXT NOT NULL, version TEXT NOT NULL,
             digest TEXT NOT NULL, status TEXT NOT NULL, content BLOB,
             UNIQUE(source, version))""")
        columns = [row[1] for row in connection.execute("PRAGMA table_info(local_documents)")]
        if "content" not in columns:
            connection.execute("ALTER TABLE local_documents ADD COLUMN content BLOB")
        connection.execute("""CREATE TABLE IF NOT EXISTS local_chunks
            (document_id INTEGER NOT NULL, location TEXT NOT NULL, text TEXT NOT NULL,
             PRIMARY KEY(document_id, location))""")
        connection.execute("""CREATE TABLE IF NOT EXISTS local_document_events
            (id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL,
             action TEXT NOT NULL, event_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        connection.execute("CREATE TABLE IF NOT EXISTS local_meta (key TEXT PRIMARY KEY)")
        legacy = store.with_suffix(".json")
        if not legacy.exists() or connection.execute("SELECT 1 FROM local_meta WHERE key='legacy_imported'").fetchone():
            return
        rows = json.loads(legacy.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("Malformed legacy document store")
        for row in rows:
            if not isinstance(row, dict) or not all(key in row for key in ("source", "version", "hash", "location", "text")):
                raise ValueError("Malformed legacy document row")
            if not all(isinstance(row[key], str) for key in ("source", "version", "hash", "location", "text")):
                raise ValueError("Malformed legacy document values")
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,50}\.txt", row["source"]):
                raise ValueError("Malformed legacy source name")
            if not re.fullmatch(r"[a-zA-Z0-9._-]{1,40}", row["version"]):
                raise ValueError("Malformed legacy version")
            if not re.fullmatch(r"[0-9a-f]{64}", row["hash"]):
                raise ValueError("Malformed legacy document hash")
            if not re.fullmatch(r"paragraph [1-9][0-9]*", row["location"]) or not row["text"].strip():
                raise ValueError("Malformed legacy paragraph")
            previous = connection.execute("SELECT id FROM local_documents WHERE source=? AND version=?",
                                          (row["source"], row["version"])).fetchone()
            if previous:
                document_id = previous[0]
                stored_hash = connection.execute("SELECT digest FROM local_documents WHERE id=?", (document_id,)).fetchone()[0]
                if stored_hash != row["hash"]:
                    raise ValueError("Conflicting legacy document hashes")
            else:
                old_ids = [item[0] for item in connection.execute(
                    "SELECT id FROM local_documents WHERE source=? AND status='CURRENT'", (row["source"],))]
                connection.execute("UPDATE local_documents SET status='OLD' WHERE source=? AND status='CURRENT'", (row["source"],))
                for old_id in old_ids:
                    record_document_event(connection, old_id, "superseded")
                document_id = connection.execute("""INSERT INTO local_documents
                    (source, version, digest, status) VALUES (?, ?, ?, 'CURRENT')""",
                    (row["source"], row["version"], row["hash"])).lastrowid
                record_document_event(connection, document_id, "imported")
            old_chunk = connection.execute("SELECT text FROM local_chunks WHERE document_id=? AND location=?",
                                           (document_id, row["location"])).fetchone()
            if old_chunk and old_chunk[0] != row["text"]:
                raise ValueError("Conflicting legacy paragraph")
            if not old_chunk:
                connection.execute("INSERT INTO local_chunks (document_id, location, text) VALUES (?, ?, ?)",
                                   (document_id, row["location"], row["text"]))
        connection.execute("INSERT INTO local_meta (key) VALUES ('legacy_imported')")


def record_document_event(connection, document_id, action):
    """Record a document state change in the same SQLite transaction."""
    connection.execute("INSERT INTO local_document_events (document_id, action) VALUES (?, ?)",
                       (document_id, action))


def read_local_documents(store, include_deleted=False):
    """Read current versions by default; inventory may request older ones."""
    if not store.exists() and not store.with_suffix(".json").exists():
        return []
    with closing(sqlite3.connect(store, timeout=3)) as connection:
        setup_documents(connection, store)
        query = """SELECT d.source, d.version, d.digest, d.status, c.location, c.text,
            d.content IS NOT NULL
            FROM local_documents d JOIN local_chunks c ON c.document_id=d.id"""
        if not include_deleted:
            query += " WHERE d.status='CURRENT'"
        query += " ORDER BY d.id, c.location"
        rows = connection.execute(query).fetchall()
    return [dict(source=row[0], version=row[1], hash=row[2], status=row[3],
                 location=row[4], text=row[5], downloadable=bool(row[6])) for row in rows]


def delete_document(store, source, version):
    """Soft-delete one local version; original SOP files cannot be touched."""
    if not store.exists():
        raise ValueError("Local document not found")
    with closing(sqlite3.connect(store, timeout=3)) as connection:
        setup_documents(connection, store)
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute("""UPDATE local_documents SET status='DELETED'
                WHERE source=? AND version=? AND status!='DELETED'""", (source, version))
            if cursor.rowcount != 1:
                raise ValueError("Local document not found or already deleted")
            document_id = connection.execute("SELECT id FROM local_documents WHERE source=? AND version=?",
                                             (source, version)).fetchone()[0]
            record_document_event(connection, document_id, "deleted")
    return {"source": source, "version": version, "status": "DELETED"}


def restore_document(store, source, version):
    """Make one deleted local version current in a single transaction."""
    if not store.exists():
        raise ValueError("Deleted local document not found")
    with closing(sqlite3.connect(store, timeout=3)) as connection:
        setup_documents(connection, store)
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            deleted = connection.execute("""SELECT id, digest, content FROM local_documents
                WHERE source=? AND version=? AND status='DELETED'""", (source, version)).fetchone()
            if not deleted:
                raise ValueError("Deleted local document not found")
            if deleted[2] is not None and hashlib.sha256(deleted[2]).hexdigest() != deleted[1]:
                raise ValueError("Stored file hash does not match")
            current = connection.execute("""SELECT id FROM local_documents
                WHERE source=? AND status='CURRENT'""", (source,)).fetchone()
            if current:
                connection.execute("UPDATE local_documents SET status='OLD' WHERE id=?", (current[0],))
                record_document_event(connection, current[0], "superseded")
            connection.execute("UPDATE local_documents SET status='CURRENT' WHERE id=?", (deleted[0],))
            record_document_event(connection, deleted[0], "restored")
    return {"source": source, "version": version, "status": "CURRENT"}


def document_history(store, source, version):
    """Return recorded state changes for one local version."""
    if not store.exists():
        raise ValueError("Local document not found")
    with closing(sqlite3.connect(store, timeout=3)) as connection:
        setup_documents(connection, store)
        row = connection.execute("SELECT id FROM local_documents WHERE source=? AND version=?",
                                 (source, version)).fetchone()
        if not row:
            raise ValueError("Local document not found")
        events = connection.execute("""SELECT action, event_at FROM local_document_events
            WHERE document_id=? ORDER BY id""", (row[0],)).fetchall()
    return [{"action": event[0], "at": event[1]} for event in events]


def read_document_file(store, source, version):
    """Return original bytes for a saved, nondeleted local version."""
    if not store.exists():
        raise ValueError("Local document not found")
    with closing(sqlite3.connect(store, timeout=3)) as connection:
        setup_documents(connection, store)
        row = connection.execute("""SELECT content, digest FROM local_documents
            WHERE source=? AND version=? AND status!='DELETED'""", (source, version)).fetchone()
    if not row or row[0] is None:
        raise ValueError("Original file is unavailable for this version")
    if hashlib.sha256(row[0]).hexdigest() != row[1]:
        raise ValueError("Stored file hash does not match")
    return row[0]


def backup_local_databases(source_dir, target_dir):
    """Copy existing SQLite stores into a new private folder consistently."""
    names = []
    for name in ("documents.db", "tickets.db", "reviews.db"):
        if (source_dir / name).is_file():
            names.append(name)
    if not names:
        raise ValueError("No local SQLite databases are available to back up")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir()
    created = []
    try:
        for name in names:
            destination_file = target_dir / name
            created.append(destination_file)
            with closing(sqlite3.connect(source_dir / name, timeout=3)) as source:
                with closing(sqlite3.connect(destination_file, timeout=3)) as destination:
                    source.backup(destination)
                    checked = destination.execute("PRAGMA integrity_check").fetchone()[0]
                    if checked != "ok":
                        raise ValueError(f"Backup integrity check failed for {name}: {checked}")
    except Exception:
        for file in created:
            file.unlink(missing_ok=True)
        try:
            target_dir.rmdir()
        except OSError:
            pass  # Preserve the original backup error if another file appeared.
        raise
    return names


def restore_local_databases(backup_dir, target_dir):
    """Restore checked SQLite snapshots into a directory without existing stores."""
    names = []
    for name in ("documents.db", "tickets.db", "reviews.db"):
        if (backup_dir / name).is_file():
            names.append(name)
    if not names:
        raise ValueError("Backup has no recognized SQLite databases")
    for name in names:
        if (target_dir / name).exists() or (target_dir / name).is_symlink():
            raise FileExistsError(f"Refusing to overwrite {name}; choose a fresh local directory")
    for name in names:
        with closing(sqlite3.connect((backup_dir / name).as_uri() + "?mode=ro", uri=True)) as source:
            checked = source.execute("PRAGMA integrity_check").fetchone()[0]
            if checked != "ok":
                raise ValueError(f"Backup integrity check failed for {name}: {checked}")
    target_dir.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for name in names:
            destination_file = target_dir / name
            created.append(destination_file)
            with closing(sqlite3.connect((backup_dir / name).as_uri() + "?mode=ro", uri=True)) as source:
                with closing(sqlite3.connect(destination_file)) as destination:
                    source.backup(destination)
                    checked = destination.execute("PRAGMA integrity_check").fetchone()[0]
                    if checked != "ok":
                        raise ValueError(f"Restored database failed integrity check: {name}")
    except (OSError, ValueError, sqlite3.DatabaseError):
        for file in created:
            file.unlink(missing_ok=True)
        raise
    return names


def document_inventory(documents):
    """Return one entry per source/version and flag exact duplicate content."""
    seen = set()
    latest = {}
    hashes = {}
    inventory = []
    for row in documents:
        key = (row["source"], row.get("version", "unknown"), row.get("hash"))
        if key in seen:
            continue
        seen.add(key)
        inventory.append(dict(source=key[0], version=key[1], hash=key[2],
                              status=row.get("status", "unknown"), changed_from_previous=False,
                              duplicate_of=hashes.get(key[2]),
                              downloadable=row.get("downloadable", False)))
        if key[0] in latest and latest[key[0]] != key[2]:
            inventory[-1]["changed_from_previous"] = True
        latest[key[0]] = key[2]
        if key[2] is not None and key[2] not in hashes:
            hashes[key[2]] = key[0]
    return inventory


def inspect_parquet(folder):
    """Print metadata and at most five sample rows from the first valid file."""
    import pyarrow.parquet as pq

    files = sorted(folder.rglob("*.parquet"))
    print("Number of Parquet files:", len(files))
    sampled = False
    for path in files:
        try:
            with pq.ParquetFile(path) as parquet:
                print(path.name, "Rows:", parquet.metadata.num_rows)
                print("Columns:", parquet.schema_arrow.names)
                if not sampled:
                    for batch in parquet.iter_batches(batch_size=5):
                        pprint(batch.to_pylist())
                        break
                    sampled = True
        except (OSError, ValueError) as error:
            print(f"Cannot inspect {path.name}: {error}")
