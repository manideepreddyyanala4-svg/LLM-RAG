"""Optional PostgreSQL storage, plus the SQLite tables for vectors and AI runs.

Nothing connects to PostgreSQL until a caller supplies a DSN.
"""
import json
import re
import sqlite3
from contextlib import closing
from datetime import date
import psycopg
from investigation import summarize_metrics

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY, source TEXT NOT NULL, version TEXT NOT NULL,
    sha256 TEXT NOT NULL, active BOOLEAN NOT NULL DEFAULT TRUE,
    UNIQUE (source, version)
);
CREATE TABLE IF NOT EXISTS chunks (
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    location TEXT NOT NULL, content TEXT NOT NULL,
    PRIMARY KEY (document_id, location)
);
CREATE TABLE IF NOT EXISTS inspections (
    date DATE NOT NULL, line TEXT NOT NULL, camera TEXT NOT NULL, model TEXT NOT NULL,
    inspected INTEGER NOT NULL CHECK (inspected >= 0),
    rejected INTEGER NOT NULL CHECK (rejected BETWEEN 0 AND inspected),
    verified_good_rejected INTEGER NOT NULL CHECK (verified_good_rejected BETWEEN 0 AND rejected),
    PRIMARY KEY (date, line, camera, model)
);
"""


def setup_database(dsn):
    """Create repeatable tables in an explicitly configured database."""
    require_dsn(dsn)
    with psycopg.connect(dsn, connect_timeout=3) as connection:
        connection.execute(SCHEMA)
        connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE")


def ingest_version(dsn, source, version, digest, chunks):
    """Insert one new version atomically; leave old versions available."""
    if not isinstance(source, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,50}\.(txt|pdf)", source):
        raise ValueError("Use a simple .txt or .pdf source filename")
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", version):
        raise ValueError("Use a simple version name")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("A SHA-256 hex hash is required")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("At least one cited chunk is required")
    locations = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Invalid chunk")
        location = chunk.get("location")
        text = chunk.get("text")
        if (not isinstance(location, str) or not re.fullmatch(r"(paragraph|page) [1-9][0-9]*", location)
                or location in locations or not isinstance(text, str) or not text.strip() or len(text) > 100_000):
            raise ValueError("Chunks need unique locations and nonempty bounded text")
        locations.add(location)
    require_dsn(dsn)
    with psycopg.connect(dsn, connect_timeout=3) as connection:
        with connection.cursor() as cursor:
            # Hold a transaction lock for this source so simultaneous versions
            # cannot both become active. Hash collisions only cause extra waiting.
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (source,))
            cursor.execute("SELECT id, sha256 FROM documents WHERE source=%s AND version=%s",
                           (source, version))
            prior = cursor.fetchone()
            if prior:
                if prior[1] != digest:
                    raise ValueError("Existing version has different content")
                return "unchanged"
            cursor.execute("UPDATE documents SET active=FALSE WHERE source=%s", (source,))
            cursor.execute("INSERT INTO documents (source, version, sha256) VALUES (%s,%s,%s) RETURNING id",
                           (source, version, digest))
            document_id = cursor.fetchone()[0]
            for chunk in chunks:
                cursor.execute("INSERT INTO chunks (document_id, location, content) VALUES (%s,%s,%s)",
                               (document_id, chunk["location"], chunk["text"]))
    return "inserted"


def list_versions(dsn):
    """Return source/version/hash/active rows without reading chunk bodies."""
    require_dsn(dsn)
    with psycopg.connect(dsn, connect_timeout=3) as connection:
        return connection.execute("SELECT source, version, sha256, active FROM documents ORDER BY source, id").fetchall()


def setup_vectors(dsn, dimensions):
    """Add vector storage only if pgvector exists and dimensions are known."""
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or not 1 <= dimensions <= 4096:
        raise ValueError("Valid embedding dimensions required")
    require_dsn(dsn)
    with psycopg.connect(dsn, connect_timeout=3) as connection:
        exists = connection.execute("SELECT 1 FROM pg_available_extensions WHERE name=%s", ("vector",)).fetchone()
        if not exists:
            raise RuntimeError("pgvector is unavailable on this server")
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        existing = connection.execute("""SELECT format_type(atttypid, atttypmod)
            FROM pg_attribute WHERE attrelid='chunks'::regclass AND attname='embedding'
            AND NOT attisdropped""").fetchone()
        expected = f"vector({dimensions})"
        if existing and existing[0] != expected:
            raise ValueError(f"Existing embedding column uses {existing[0]}, expected {expected}")
        if not existing:
            # SQL parameters cannot replace a type size; dimensions was range-checked.
            connection.execute(f"ALTER TABLE chunks ADD COLUMN embedding {expected}")


def ingest_inspections(dsn, rows):
    """Validate then insert small operational records without duplicates."""
    require_dsn(dsn)
    summarize_metrics(rows)
    with psycopg.connect(dsn, connect_timeout=3) as connection:
        for row in rows:
            identity = (date.fromisoformat(row["date"]), row["line"], row["camera"], row["model"])
            counts = (int(row["inspected"]), int(row["rejected"]), int(row["verified_good_rejected"]))
            cursor = connection.execute("""INSERT INTO inspections
                (date, line, camera, model, inspected, rejected, verified_good_rejected)
                VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                identity + counts)
            if cursor.rowcount == 0:
                existing = connection.execute("""SELECT inspected, rejected, verified_good_rejected
                    FROM inspections WHERE date=%s AND line=%s AND camera=%s AND model=%s""",
                    identity).fetchone()
                if existing != counts:
                    raise ValueError("Existing inspection record has different counts")


def delete_inactive_version(dsn, source, version):
    """Delete exactly one inactive version and its chunks in one transaction."""
    if not source or not version:
        raise ValueError("Exact source and version are required")
    require_dsn(dsn)
    with psycopg.connect(dsn, connect_timeout=3) as connection:
        cursor = connection.execute("DELETE FROM documents WHERE source=%s AND version=%s AND active=FALSE",
                                    (source, version))
        if cursor.rowcount != 1:
            raise ValueError("Inactive version not found; active versions cannot be deleted")


def require_dsn(dsn):
    """Reject empty settings so psycopg cannot use an implicit local login."""
    if not isinstance(dsn, str) or not dsn.strip():
        raise ValueError("An explicit PostgreSQL DSN is required")


# SQLite stores for the AI layer. The default lab setup uses SQLite; PostgreSQL above is optional.
def embedding_cache(path, model_key):
    """Read vectors for exactly one model digest from the document database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=3)) as connection:
        with connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS embeddings
                (model_key TEXT, source_id TEXT, dimensions INTEGER NOT NULL,
                 vector_json TEXT NOT NULL, PRIMARY KEY(model_key, source_id))""")
        rows = connection.execute("SELECT source_id, dimensions, vector_json FROM embeddings WHERE model_key=?",
                                  (model_key,)).fetchall()
    result = {}
    for source_id, dimensions, vector in rows:
        values = json.loads(vector)
        if not isinstance(values, list) or len(values) != dimensions:
            raise ValueError("Stored embedding dimensions are corrupt; reindex the documents")
        result[source_id] = values
    return result


def save_embeddings(path, model_key, rows, replace=False):
    """Commit a complete batch atomically; reindex can remove obsolete vectors."""
    with closing(sqlite3.connect(path, timeout=3)) as connection:
        with connection:
            if replace:
                connection.execute("DELETE FROM embeddings")
            for source_id, vector in rows.items():
                connection.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                                   (model_key, source_id, len(vector), json.dumps(vector, allow_nan=False)))


def save_ai_run(path, result):
    """Persist the actual answer, checks, tool trace, models and timings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=3)) as connection:
        with connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS ai_runs
                (id INTEGER PRIMARY KEY, result_json TEXT NOT NULL,
                 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
            cursor = connection.execute("INSERT INTO ai_runs (result_json) VALUES (?)",
                                        (json.dumps(result, ensure_ascii=False, allow_nan=False),))
    return cursor.lastrowid


def list_ai_runs(path):
    """Read the latest twenty saved AI runs, including unsuccessful runs."""
    if not path.exists():
        return []
    with closing(sqlite3.connect(path, timeout=3)) as connection:
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='ai_runs'").fetchone():
            return []
        rows = connection.execute("SELECT id, result_json, created_at FROM ai_runs ORDER BY id DESC LIMIT 20").fetchall()
    return [dict(json.loads(row[1]), run_id=row[0], at=row[2]) for row in rows]
