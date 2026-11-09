"""Run EvidenceOps from a terminal: python main.py <command> [options].

Each command is one small function below. run_command() picks the function,
and main() reads the options and turns errors into a short message.
"""
import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from data_io import DATA, ROOT, LOCAL, backup_local_databases, restore_local_databases, inspect_parquet, load_documents, read_csv
from retrieval import search_documents, current_evidence
from evaluation import evaluate_questions, print_evaluation, compare_searches, read_verified_questions, run_ai_evaluation
from investigation import investigation_report
from workflow import run_workflow
from rag import answer_question, run_agent
from steel_faults import download_dataset, load_dataset, summarize, steel_documents, STEEL_DIR, CITATION

COMMANDS = ["search", "inspect", "labels", "evaluate", "investigate", "compare", "demo", "workflow",
            "backup", "restore", "ask", "agent", "aieval", "steel-download", "steel"]


def create_demo_data(folder):
    """Create the tiny synthetic Mode A example, only if none of its files exist."""
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
        # "x" mode creates a new file and fails if it already exists: never overwrite.
        with (folder / name).open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
    return len(files)


def print_answer(result):
    """Print a generated answer with its checked citations and real timings."""
    print("Status:", result["status"])
    print("Answer:", result["answer"])
    for claim in result["claims"]:
        print(f"  claim [{claim['citation']}] {claim['text']}")
        print(f"    verified quote: {claim['quote']}")
    for claim in result.get("withheld_claims", []):
        print(f"  WITHHELD [{claim['citation']}] {claim['text']}")
    for row in result.get("excluded_evidence", []):
        print(f"  EXCLUDED {row['source']} {row['location']}: {row['reason']}")
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


def ask_text(given, prompt):
    """Use the --question value if given, otherwise ask the person to type one."""
    if given is not None:
        return given
    return input(prompt)


def command_demo(args):
    count = create_demo_data(DATA)
    print(f"Created {count} explicitly synthetic example files in {DATA}")
    print("The example question label has not been human verified.")


def command_ask(args):
    question = ask_text(args.question, "Question: ")
    if args.steel:
        print_answer(answer_question(question, args.method, steel_documents()))
    else:
        print_answer(answer_question(question, args.method))


def command_agent(args):
    request = ask_text(args.question, "Investigation request: ")
    result = run_agent(request, args.allow_draft)
    print(f"Steps used: {result['steps_used']} of {result['step_limit']} ({result['stopped_because']})")
    number = 0
    for step in result["steps"]:
        number += 1
        if step["ok"]:
            state = "ok"
        else:
            state = "refused/failed"
        print(f"  {number}. {step['tool']} {json.dumps(step['arguments'], ensure_ascii=False)} -> {state}")
        print("     ", json.dumps(step["output"], ensure_ascii=False)[:300])
    if result["ticket"]:
        print("Pending draft:", result["ticket"], "(a human must approve it)")
    print_answer(result)


def command_aieval(args):
    report = run_ai_evaluation(LOCAL)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if report.get("steel_skipped"):
        print("Steel dataset cases skipped:", report["steel_skipped"])
    if report["errors"]:
        print("Errors:", json.dumps(report["errors"], ensure_ascii=False))
    print("\nSaved to:", LOCAL / "ai_evaluation.json")


def command_steel_download(args):
    saved = download_dataset(STEEL_DIR)
    print("Downloaded and checked:", ", ".join(saved))
    print("Saved unchanged in:", STEEL_DIR)
    print("Please cite:", CITATION)


def command_steel(args):
    summary = summarize(load_dataset(STEEL_DIR))
    print(f"UCI Steel Plates Faults: {summary['records']} records, {summary['columns']} columns, "
          f"missing values: {summary['missing_values']}")
    print("Fault categories (calculated in Python):")
    for category in summary["categories"]:
        print(f"  {category['name']:13s} {category['records']:5d}  {category['percent']:6.2f}%")
    print(f"Steel type A300: {summary['steel_a300']}, A400: {summary['steel_a400']}")
    print("Source:", summary["source"])
    print("License:", summary["license"])
    print("Please cite:", summary["citation"])


def command_backup(args):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = LOCAL / "backups" / stamp
    names = backup_local_databases(LOCAL, target)
    print("Private backup:", target)
    print("SQLite files:", ", ".join(names))


def command_restore(args):
    if not args.backup_dir:
        raise ValueError("Provide --backup-dir and use a fresh EVIDENCEOPS_LOCAL_DIR")
    names = restore_local_databases(Path(args.backup_dir).resolve(), LOCAL)
    print("Restored SQLite files into:", LOCAL)
    print("SQLite files:", ", ".join(names))


def command_workflow(args, documents):
    question = ask_text(args.question, "Question: ")
    result = run_workflow(question, args.kind)
    print(result["report"])
    print("Trace:", " -> ".join(result["trace"]))
    if result.get("ticket"):
        print("Pending ticket:", result["ticket"])


def command_compare(args, documents):
    questions = read_csv(DATA / "questions.csv", ["question_id", "question"])
    reviewed = read_verified_questions(LOCAL / "reviews.db", DATA / "questions.csv")
    rows = compare_searches(documents, questions, LOCAL / "retrieval_experiments.jsonl", reviewed)
    for row in rows:
        sources = []
        for result in row["results"]:
            sources.append(result["source"])
        if row["failure"]:
            status = row["failure"]
        elif row["reviewed"]:
            status = "reviewed hit"
        else:
            status = "unreviewed"
        print(row["id"], row["method"], sources, row["elapsed_ms"], "ms", status)
    print("Saved raw ranks, scores, and timings in data/local/retrieval_experiments.jsonl")
    print("Only saved reviews can identify source retrieval misses; unreviewed rows are unscored.")


def command_labels(args, documents):
    questions = read_csv(DATA / "questions.csv", ["question_id", "question", "expected_source"])
    for row in questions:
        print(row["question_id"], row["question"], "=>", row["expected_source"])
    print("These supplied labels have no recorded human verification.")
    print("Source options: sop_v1.txt, sop_v2_en.txt, sop_v2_zh.txt, metrics.csv, deployments.csv")


def command_evaluate(args, documents):
    questions_file = DATA / "questions.csv"
    # Check the file has the needed columns first, so a broken file gives a clear error.
    read_csv(questions_file, ["question_id", "question", "expected_source"])
    reviewed = read_verified_questions(LOCAL / "reviews.db", questions_file)
    if not reviewed:
        raise ValueError("No saved reviews for the current questions.csv")
    report = evaluate_questions(documents, reviewed, verified=True)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_evaluation(report)


def command_search(args, documents):
    """Keyword search. With --question it runs once; without, it keeps asking until /quit."""
    print(f"Keyword search: {len(documents)} evidence chunks. Type /quit to leave.")
    while True:
        question = ask_text(args.question, "Question: ")
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


def run_command(args):
    """Call the function for the chosen command."""
    action = args.action
    # These commands do not need the Mode A evidence files.
    if action == "demo":
        command_demo(args)
    elif action == "ask":
        command_ask(args)
    elif action == "agent":
        command_agent(args)
    elif action == "aieval":
        command_aieval(args)
    elif action == "steel-download":
        command_steel_download(args)
    elif action == "steel":
        command_steel(args)
    elif action == "inspect":
        inspect_parquet(ROOT / "data" / "public")
    elif action == "backup":
        command_backup(args)
    elif action == "restore":
        command_restore(args)
    else:
        # These commands read the Mode A evidence first.
        documents = load_documents(DATA, args.include_csv, LOCAL / "documents.db")
        if action == "workflow":
            command_workflow(args, documents)
        elif action == "compare":
            command_compare(args, documents)
        elif action == "labels":
            command_labels(args, documents)
        elif action == "evaluate":
            command_evaluate(args, documents)
        elif action == "investigate":
            print(investigation_report(DATA, current_evidence(documents)))
        else:
            command_search(args, documents)


def main():
    """Read the command line, run one command, and return an exit code (0 = success)."""
    parser = argparse.ArgumentParser(description="EvidenceOps terminal")
    parser.add_argument("action", nargs="?", default="search", choices=COMMANDS)
    parser.add_argument("--question", help="Ask once instead of opening the question loop")
    parser.add_argument("--kind", choices=["inspection", "deployment", "sop_conflict"],
                        default="inspection", help="Choose a deterministic workflow")
    parser.add_argument("--include-csv", action="store_true", help="Search CSV records too")
    parser.add_argument("--method", choices=["keyword", "bm25", "semantic", "hybrid"], default="hybrid",
                        help="Retrieval method for the ask command")
    parser.add_argument("--steel", action="store_true",
                        help="Ask about the real UCI steel dataset instead of the synthetic SOPs")
    parser.add_argument("--allow-draft", action="store_true",
                        help="Let the agent prepare a pending incident draft")
    parser.add_argument("--json", action="store_true", help="Print a reviewed evaluation as JSON")
    parser.add_argument("--backup-dir", help="Existing backup directory for a non-overwriting restore")
    args = parser.parse_args()
    try:
        run_command(args)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, sqlite3.DatabaseError) as error:
        print(f"Cannot complete action: {error}")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
