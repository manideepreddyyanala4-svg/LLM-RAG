"""Run EvidenceOps offline from a terminal."""
import argparse
import json
import sqlite3
import sys
from datetime import datetime
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


def main():
    """Read command options, load evidence and run the selected offline action."""
    parser = argparse.ArgumentParser(description="EvidenceOps laptop Version 1")
    parser.add_argument("action", nargs="?", default="search",
                        choices=["search", "inspect", "labels", "evaluate", "investigate", "compare", "demo",
                                 "workflow", "backup", "restore"])
    parser.add_argument("--question", help="Ask once instead of opening the question loop")
    parser.add_argument("--kind", choices=["inspection", "deployment", "sop_conflict"],
                        default="inspection", help="Choose a deterministic workflow")
    parser.add_argument("--include-csv", action="store_true", help="Search CSV records too")
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
