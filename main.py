"""Run EvidenceOps offline from a terminal."""
import argparse
import json
import sqlite3
import sys
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from data_io import DATA, ROOT, LOCAL, backup_local_databases, restore_local_databases, inspect_parquet, load_documents, read_csv
from retrieval import search_documents, current_evidence
from evaluation import evaluate_questions, print_evaluation, compare_searches, read_verified_questions
from investigation import investigation_report
from workflow import run_workflow


def create_demo_data(folder):
    """Create a tiny synthetic example only when none of its files exist."""
    files = {
        "sop_v1.txt": "DOCUMENT ID: demo-v1\nSTATUS: OLD\n\nHistorical Line B threshold was 0.40.\n",
        "sop_v2_en.txt": "DOCUMENT ID: demo-v2\nSTATUS: ACTIVE\n\nCurrent Line B threshold is 0.50. Check calibration records before changing settings.\n",
        "sop_v2_zh.txt": "文件編號: demo-v2-zh\n狀態: 有效\n\n目前 B 線閾值為 0.50。請檢查校準紀錄。\n",
        "metrics.csv": "date,line,camera,model,inspected,rejected,verified_good_rejected\n2026-01-01,B,C1,v3,100,5,1\n2026-01-02,B,C1,v4,100,7,1\n",
        "deployments.csv": "deployment_id,date,line,model,defect_score_threshold,image_size,status\ndemo-v3,2026-01-01,B,v3,0.40,640,historical\ndemo-v4,2026-01-02,B,v4,0.50,640,current\n",
        "questions.csv": "question_id,question,expected_source\nDEMO1,Where are calibration records mentioned?,sop_v2_en.txt\n",
    }
    for name in files:
        if (folder / name).exists():
            raise ValueError(f"Demo creation stopped: {name} already exists")
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        with (folder / name).open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
    return len(files)


def print_answer(result):
    """Print the generated answer with its checked citations and real timings."""
    print("Status:", result["status"])
    print("Answer:", result["answer"])
    for claim in result["claims"]:
        print(f"  claim [{claim['citation']}] {claim['text']}")
        print(f"    verified quote: {claim['quote']}")
    for claim in result.get("withheld_claims", []):
        print(f"  WITHHELD [{claim['citation']}] {claim['text']}")
    if result["citation_errors"]:
        print("Citation problems:", "; ".join(result["citation_errors"]))
    if result["unsupported_claims"]:
        print("Claims the checker rejected:", result["unsupported_claims"])
    print("Evidence used:")
    for row in result["evidence"]:
        print(f"  [{row['citation_id']}] {row['source']} {row['location']} ({row.get('version')})")
    print(f"Timings ms: retrieval={result.get('retrieval_ms')} index={result.get('index_ms')} "
          f"generation={result['inference_ms']} citation_check={result['verification_ms']}")
    print("Generation model:", result["model"], "| embedding model:", result.get("embedding_model"),
          "| dimensions:", result.get("dimensions"))
    print("Note:", result["verification"])


# Deterministic test cases written for this lab. These are NOT human-reviewed
# benchmark labels: each one only names evidence that is present in the files.
AI_TEST_CASES = [
    {"id": "EN-THRESHOLD", "question": "What is the current Line B threshold?",
     "expect_source": "sop_v2_en.txt", "language": "English"},
    {"id": "ZH-THRESHOLD", "question": "目前 B 線的閾值是多少？",
     "expect_source": "sop_v2_zh.txt", "language": "Traditional Chinese"},
    {"id": "EN-CALIBRATION", "question": "What must be checked before changing settings?",
     "expect_source": "sop_v2_en.txt", "language": "English"},
    {"id": "ZH-CALIBRATION", "question": "更改設定前需要檢查什麼？",
     "expect_source": "sop_v2_zh.txt", "language": "Traditional Chinese"},
    {"id": "EN-ABSENT", "question": "Who approved the Line C deployment in 2019?",
     "expect_source": None, "language": "English"},
    {"id": "ZH-ABSENT", "question": "C 線在 2019 年由誰核准部署？",
     "expect_source": None, "language": "Traditional Chinese"},
]


def hardware_facts():
    """Record the lab machine and GPU only from what the system actually reports."""
    facts = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        facts["total_ram_kb"] = int(Path("/proc/meminfo").read_text().split()[1])
    except (OSError, ValueError, IndexError):
        facts["total_ram_kb"] = None
    gpu = shutil.which("nvidia-smi")
    facts["gpu"] = None
    if gpu:
        try:
            probe = subprocess.run([gpu, "--query-gpu=name,memory.total,driver_version",
                                    "--format=csv,noheader"], capture_output=True, timeout=15)
            if probe.returncode == 0:
                facts["gpu"] = probe.stdout.decode("utf-8", errors="replace").strip()
        except (OSError, subprocess.TimeoutExpired):
            facts["gpu"] = None
    return facts


def run_ai_evaluation():
    """Run every method and every test case against the real local models."""
    from rag import answer_question, compare_methods, run_agent, ai_status, evidence_documents
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "hardware": hardware_facts(), "ai_status": ai_status(),
              "labels": "Deterministic lab test cases. Not human-reviewed benchmark labels. "
                        "expect_source only records which file holds the evidence.",
              "retrieval": [], "answers": [], "agent": None, "errors": []}
    documents = evidence_documents()
    for case in AI_TEST_CASES:
        try:
            comparison = compare_methods(case["question"], documents)
        except (RuntimeError, ValueError, OSError) as error:
            report["errors"].append({"case": case["id"], "stage": "retrieval", "error": str(error)})
            continue
        for row in comparison["comparison"]:
            sources = [hit["source"] for hit in row["ranking"]]
            found = None
            if case["expect_source"]:
                found = case["expect_source"] in sources
            report["retrieval"].append({
                "case": case["id"], "language": case["language"], "method": row["method"],
                "ranked_sources": sources, "source_ids": row["source_ids"],
                "expected_source_in_top3": found, "retrieval_ms": row["retrieval_ms"],
                "index_ms": row["index_ms"]})
    for case in AI_TEST_CASES:
        try:
            answer = answer_question(case["question"], "hybrid")
        except (RuntimeError, ValueError, OSError) as error:
            report["errors"].append({"case": case["id"], "stage": "generation", "error": str(error)})
            continue
        report["answers"].append({
            "case": case["id"], "language": case["language"], "question": case["question"],
            "status": answer["status"], "answer": answer["answer"],
            "claims": answer["claims"], "withheld_claims": answer.get("withheld_claims", []),
            "cited_source_ids": [row.get("source_id") for row in answer["evidence"]],
            "citation_errors": answer["citation_errors"],
            "unsupported_claims": answer["unsupported_claims"],
            "evidence_sufficient": answer["status"] != "insufficient",
            "expected_evidence_present": case["expect_source"] is not None,
            "inference_ms": answer["inference_ms"], "verification_ms": answer["verification_ms"],
            "retrieval_ms": answer["retrieval_ms"], "run_id": answer["run_id"]})
    report["repeatability"] = []
    for case in AI_TEST_CASES[:4]:
        statuses = []
        for _ in range(3):
            try:
                statuses.append(answer_question(case["question"], "hybrid")["status"])
            except (RuntimeError, ValueError, OSError) as error:
                statuses.append(f"error: {error}")
        report["repeatability"].append({"case": case["id"], "runs": 3, "statuses": statuses,
                                        "identical": len(set(statuses)) == 1})
    try:
        agent = run_agent("Investigate the Line B rejection increase against the SOP", False)
        report["agent"] = {"steps_used": agent["steps_used"], "step_limit": agent["step_limit"],
                           "stopped_because": agent["stopped_because"],
                           "tools_approved": agent["tools_approved"],
                           "calls": [{"tool": step["tool"], "arguments": step["arguments"],
                                      "ok": step["ok"],
                                      "output": str(step["output"])[:300]} for step in agent["steps"]],
                           "draft_created": agent["ticket"], "summary_status": agent["status"],
                           "planning_ms": agent["planning_ms"]}
    except (RuntimeError, ValueError, OSError) as error:
        report["errors"].append({"case": "agent", "stage": "agent", "error": str(error)})
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / "ai_evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    return report


def main():
    """Read command options, load evidence and run the selected offline action."""
    parser = argparse.ArgumentParser(description="EvidenceOps laptop Version 1")
    parser.add_argument("action", nargs="?", default="search",
                        choices=["search", "inspect", "labels", "evaluate", "investigate", "compare", "demo",
                                 "workflow", "backup", "restore", "ask", "agent", "aieval"])
    parser.add_argument("--question", help="Ask once instead of opening the question loop")
    parser.add_argument("--kind", choices=["inspection", "deployment", "sop_conflict"],
                        default="inspection", help="Choose a deterministic workflow")
    parser.add_argument("--include-csv", action="store_true", help="Search CSV records too")
    parser.add_argument("--method", choices=["keyword", "bm25", "semantic", "hybrid"], default="hybrid",
                        help="Retrieval method for the ask action")
    parser.add_argument("--allow-draft", action="store_true",
                        help="Let the agent prepare a pending incident draft")
    parser.add_argument("--json", action="store_true", help="Print a reviewed evaluation as JSON")
    parser.add_argument("--backup-dir", help="Existing backup directory for a non-overwriting restore")
    parser.add_argument("--labels-verified", action="store_true",
                        help="Legacy flag; saved per-question reviews are still required")
    args = parser.parse_args()
    try:
        if args.action == "demo":
            count = create_demo_data(DATA)
            print(f"Created {count} explicitly synthetic example files in {DATA}")
            print("The example question label has not been human verified.")
            return 0
        if args.action in ("ask", "agent", "aieval"):
            from rag import answer_question, run_agent
            if args.action == "ask":
                question = args.question or input("Question: ")
                print_answer(answer_question(question, args.method))
                return 0
            if args.action == "agent":
                request = args.question or input("Investigation request: ")
                result = run_agent(request, args.allow_draft)
                print(f"Steps used: {result['steps_used']} of {result['step_limit']} "
                      f"({result['stopped_because']})")
                for number, step in enumerate(result["steps"], 1):
                    state = "ok" if step["ok"] else "refused/failed"
                    print(f"  {number}. {step['tool']} {json.dumps(step['arguments'], ensure_ascii=False)} -> {state}")
                    print("     ", json.dumps(step["output"], ensure_ascii=False)[:300])
                if result["ticket"]:
                    print("Pending draft:", result["ticket"], "(a human must approve it)")
                print_answer(result)
                return 0
            report = run_ai_evaluation()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print("\nSaved to:", LOCAL / "ai_evaluation.json")
            return 0
        if args.action == "inspect":
            inspect_parquet(ROOT / "data" / "public")
            return 0
        if args.action == "backup":
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            target = LOCAL / "backups" / stamp
            names = backup_local_databases(LOCAL, target)
            print("Private backup:", target)
            print("SQLite files:", ", ".join(names))
            return 0
        if args.action == "restore":
            if not args.backup_dir:
                raise ValueError("Provide --backup-dir and use a fresh EVIDENCEOPS_LOCAL_DIR")
            names = restore_local_databases(Path(args.backup_dir).resolve(), LOCAL)
            print("Restored SQLite files into:", LOCAL)
            print("SQLite files:", ", ".join(names))
            return 0
        documents = load_documents(DATA, args.include_csv, LOCAL / "documents.db")
        if args.action == "workflow":
            question = args.question or input("Question: ")
            result = run_workflow(question, args.kind)
            print(result["report"])
            print("Trace:", " -> ".join(result["trace"]))
            if result.get("ticket"):
                print("Pending ticket:", result["ticket"])
            return 0
        if args.action == "compare":
            questions = read_csv(DATA / "questions.csv", ["question_id", "question"])
            reviewed = read_verified_questions(LOCAL / "reviews.db", DATA / "questions.csv")
            rows = compare_searches(documents, questions, LOCAL / "retrieval_experiments.jsonl", reviewed)
            for row in rows:
                sources = [result["source"] for result in row["results"]]
                status = row["failure"] or ("reviewed hit" if row["reviewed"] else "unreviewed")
                print(row["id"], row["method"], sources, row["elapsed_ms"], "ms", status)
            print("Saved raw ranks, scores, and timings in data/local/retrieval_experiments.jsonl")
            print("Only saved reviews can identify source retrieval misses; unreviewed rows are unscored.")
            return 0
        if args.action in ("labels", "evaluate"):
            questions_file = DATA / "questions.csv"
            questions = read_csv(questions_file,
                                 ["question_id", "question", "expected_source"])
            if args.action == "labels":
                for row in questions:
                    print(row["question_id"], row["question"], "=>", row["expected_source"])
                print("These supplied labels have no recorded human verification.")
                print("Source options: sop_v1.txt, sop_v2_en.txt, sop_v2_zh.txt, metrics.csv, deployments.csv")
            else:
                reviewed = read_verified_questions(LOCAL / "reviews.db", questions_file)
                if not reviewed:
                    raise ValueError("No saved reviews for the current questions.csv")
                report = evaluate_questions(documents, reviewed, verified=True)
                if args.json:
                    print(json.dumps(report, ensure_ascii=False, indent=2))
                else:
                    print_evaluation(report)
            return 0
        if args.action == "investigate":
            print(investigation_report(DATA, current_evidence(documents)))
            return 0
        print(f"Keyword search: {len(documents)} evidence chunks. Type /quit to leave.")
        while True:
            question = args.question
            if question is None:
                question = input("Question: ")
            if question.strip() == "/quit":
                break
            if not question.strip():
                print("Please enter a non-empty question.")
            else:
                results = search_documents(documents, question)
                if not results:
                    print("No matching evidence. English keyword overlap only.")
                for result in results:
                    print(f"[{result['source']} {result['location']}] Score: {result['score']}")
                    print("Matched words:", ", ".join(result["matched"]))
                    print(result["text"])
            if args.question is not None:
                break
        return 0
    except (OSError, ValueError, ImportError, sqlite3.DatabaseError) as error:
        print(f"Cannot complete action: {error}")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
