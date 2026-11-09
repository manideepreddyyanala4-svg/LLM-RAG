"""Real MCP stdio server exposing four bounded, token-protected tools."""
import os
import sqlite3
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from data_io import DATA, LOCAL, load_documents, read_csv
from retrieval import search_bilingual, current_evidence
from investigation import investigation_report, prepare_ticket, summarize_metrics

server = MCPServer("EvidenceOps local tools")


def authorize():
    """Require a token in the trusted local server environment."""
    expected = os.environ.get("EVIDENCEOPS_TOKEN", "")
    if not expected:
        raise ToolError("Server access is not configured")


def authorize_operator():
    """Allow ticket drafts only in a trusted operator-mode MCP process."""
    token = os.environ.get("EVIDENCEOPS_OPERATOR_TOKEN", "")
    if os.environ.get("EVIDENCEOPS_MCP_ROLE", "viewer") != "operator" or not token:
        raise ToolError("MCP operator role is not configured")
    if token in (os.environ.get("EVIDENCEOPS_TOKEN"), os.environ.get("EVIDENCEOPS_APPROVAL_TOKEN")):
        raise ToolError("MCP operator token must be separate")


@server.tool()
def search_approved_documents(question: str, include_history: bool = False) -> list[dict]:
    """Search current SOPs/uploads; include older versions only when asked."""
    authorize()
    if not question.strip() or len(question) > 1000:
        raise ToolError("Question must contain 1 to 1000 characters")
    try:
        documents = load_documents(DATA, local_file=LOCAL / "documents.db",
                                   include_deleted=include_history)
        if include_history:
            documents = [row for row in documents if row.get("status") != "DELETED"]
        else:
            documents = current_evidence(documents)
        return search_bilingual(documents, question)
    except (OSError, ValueError, sqlite3.DatabaseError) as error:
        raise ToolError("Evidence unavailable") from error


@server.tool()
def read_inspection_metrics(line: str) -> dict:
    """Return calculated totals for exactly one line."""
    authorize()
    if line not in ("A", "B"):
        raise ToolError("Line must be A or B")
    try:
        rows = read_csv(DATA / "metrics.csv", ["date", "line", "camera", "model",
                        "inspected", "rejected", "verified_good_rejected"])
        groups = summarize_metrics(rows)
    except (OSError, ValueError) as error:
        raise ToolError("Metrics unavailable") from error
    return {model: values for (record_line, model), values in groups.items() if record_line == line}


@server.tool()
def compare_model_deployments(line: str) -> list[dict]:
    """Return actual deployment records for the selected line."""
    authorize()
    if line not in ("A", "B"):
        raise ToolError("Line must be A or B")
    try:
        rows = read_csv(DATA / "deployments.csv", ["deployment_id", "date", "line",
                        "model", "defect_score_threshold", "image_size", "status"])
    except (OSError, ValueError) as error:
        raise ToolError("Deployments unavailable") from error
    return [row for row in rows if row["line"] == line]


@server.tool()
def prepare_incident_ticket(question: str) -> dict:
    """Create a prepared draft; this tool cannot approve or close tickets."""
    authorize()
    authorize_operator()
    if not question.strip() or len(question) > 1000:
        raise ToolError("Question must contain 1 to 1000 characters")
    try:
        report = investigation_report(DATA, current_evidence(load_documents(DATA)))
        return prepare_ticket(LOCAL / "tickets.db", question, report)
    except (OSError, ValueError, sqlite3.DatabaseError) as error:
        raise ToolError("Cannot prepare ticket") from error


if __name__ == "__main__":
    server.run(transport="stdio")
