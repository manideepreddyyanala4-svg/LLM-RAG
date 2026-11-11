"""Run isolated PostgreSQL/pgvector checks with synthetic data only.

Set EVIDENCEOPS_TEST_DATABASE_URL explicitly. This creates and removes a
new, randomly named test schema; it never changes the application's tables.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from uuid import uuid4

import psycopg

import database


def add_version(dsn, source, version):
    """Add one synthetic version to exercise the real database adapter."""
    content = f"Synthetic note for {source} {version}"
    digest = hashlib.sha256(content.encode()).hexdigest()
    chunks = [{"location": "paragraph 1", "text": content}]
    return database.ingest_version(dsn, source, version, digest, chunks)


def check_versions(dsn):
    """Check version conflicts and safe deletion with synthetic documents."""
    assert add_version(dsn, "smoke.txt", "v1") == "inserted"
    assert add_version(dsn, "smoke.txt", "v1") == "unchanged"
    try:
        database.ingest_version(dsn, "smoke.txt", "v1", "f" * 64,
                                [{"location": "paragraph 1", "text": "Different"}])
    except ValueError:
        pass
    else:
        raise AssertionError("Changed content for an existing version was accepted")
    assert add_version(dsn, "smoke.txt", "v2") == "inserted"
    try:
        database.delete_inactive_version(dsn, "smoke.txt", "v2")
    except ValueError:
        pass
    else:
        raise AssertionError("The active version was deleted")
    with psycopg.connect(dsn, connect_timeout=3) as connection:
        connection.execute("ALTER TABLE chunks ADD CONSTRAINT smoke_no_forbidden "
                           "CHECK (content <> 'forbidden')")
    try:
        database.ingest_version(dsn, "smoke.txt", "broken", "a" * 64,
                                [{"location": "paragraph 1", "text": "forbidden"}])
    except psycopg.errors.CheckViolation:
        pass
    else:
        raise AssertionError("The expected database write failure did not occur")
    versions = database.list_versions(dsn)
    current = []
    for row in versions:
        if row[0] == "smoke.txt" and row[3]:  # row[3] is the "active" flag
            current.append(row)
    assert len(current) == 1 and current[0][1] == "v2"
    database.delete_inactive_version(dsn, "smoke.txt", "v1")


def check_concurrency(dsn):
    """Check that competing writes leave one active version per source."""
    for number in range(10):
        source = f"parallel_{number}.txt"
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(add_version, dsn, source, "v1"),
                    pool.submit(add_version, dsn, source, "v2")]
            for job in jobs:
                assert job.result() == "inserted"
        with psycopg.connect(dsn, connect_timeout=3) as connection:
            counts = connection.execute("SELECT count(*), count(*) FILTER (WHERE active) "
                                        "FROM documents WHERE source=%s", (source,)).fetchone()
        assert counts == (2, 1), f"Concurrent versions failed for {source}: {counts}"


def check_inspections_and_vectors(dsn):
    """Check count conflicts and a similarity query with synthetic vectors."""
    row = {"date": "2026-01-01", "line": "B", "camera": "C1", "model": "v4",
           "inspected": "10", "rejected": "2", "verified_good_rejected": "1"}
    database.ingest_inspections(dsn, [row])
    database.ingest_inspections(dsn, [row])
    try:
        database.ingest_inspections(dsn, [dict(row, rejected="3")])
    except ValueError:
        pass
    else:
        raise AssertionError("Conflicting inspection counts were accepted")

    with psycopg.connect(dsn, connect_timeout=3) as connection:
        connection.execute("UPDATE chunks SET embedding=%s::vector WHERE content=%s",
                           ("[1,0,0]", "Synthetic note for smoke.txt v2"))
        nearest = connection.execute("SELECT content FROM chunks WHERE embedding IS NOT NULL "
                                     "ORDER BY embedding <=> %s::vector LIMIT 1",
                                     ("[0.9,0.1,0]",)).fetchone()[0]
        extension = connection.execute("SELECT extversion FROM pg_extension "
                                       "WHERE extname='vector'").fetchone()[0]
        inspections = connection.execute("SELECT count(*) FROM inspections").fetchone()[0]
    assert nearest == "Synthetic note for smoke.txt v2"
    assert inspections == 1
    try:
        database.setup_vectors(dsn, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("A different vector dimension was accepted")
    return extension


def run_checks(dsn):
    """Create one test schema, run all checks, then remove that schema."""
    # SQL parameters cannot name schemas. A fixed prefix and UUID hex keep
    # this test-only SQL identifier safe and separate from application data.
    schema = "evidenceops_test_" + uuid4().hex
    with psycopg.connect(dsn, connect_timeout=3) as connection:
        connection.execute(f"CREATE SCHEMA {schema}")
    try:
        test_dsn = psycopg.conninfo.make_conninfo(dsn, options=f"-c search_path={schema},public")
        with psycopg.connect(test_dsn, connect_timeout=3) as connection:
            connection.execute("CREATE TABLE documents (id BIGSERIAL PRIMARY KEY, "
                               "source TEXT NOT NULL, version TEXT NOT NULL, "
                               "sha256 TEXT NOT NULL, UNIQUE (source, version))")
        database.setup_database(test_dsn)
        database.setup_database(test_dsn)
        database.setup_vectors(test_dsn, 3)
        check_versions(test_dsn)
        check_concurrency(test_dsn)
        extension = check_inspections_and_vectors(test_dsn)
        print(f"PostgreSQL smoke passed: pgvector {extension}; migration, versions, "
              "rollback, 10 concurrent source pairs, inspections, and synthetic vector query")
    finally:
        with psycopg.connect(dsn, connect_timeout=3) as connection:
            connection.execute(f"DROP SCHEMA {schema} CASCADE")


if __name__ == "__main__":
    database_url = os.environ.get("EVIDENCEOPS_TEST_DATABASE_URL")
    if not database_url:
        raise SystemExit("Set EVIDENCEOPS_TEST_DATABASE_URL to an isolated test database")
    run_checks(database_url)
