"""Mode B: analyse the real UCI Steel Plates Faults dataset with plain Python.

Source:   UCI Machine Learning Repository, dataset 198, "Steel Plates Faults".
Creators: M. Buscema, S. Terzi and W. Tastle (2010).
License:  Creative Commons Attribution 4.0 International (CC BY 4.0).
DOI:      https://doi.org/10.24432/C5J88N

The two raw files are downloaded once from UCI, checked against fixed SHA-256
hashes, and stored unchanged in data/steel_faults/ (ignored by Git). Every
number below is calculated here in Python. The AI model never calculates
anything for this dataset: it may only quote the summary sentences this file
writes, and the quote check in ollama_client.py makes sure of that.

This dataset is unrelated to the synthetic Line B example (Mode A).
"""
import hashlib
import io
import os
import zipfile
from math import isfinite
from pathlib import Path
from statistics import mean, median
from urllib.request import Request, build_opener, ProxyHandler
from data_io import ROOT

DATASET_PAGE = "https://archive.ics.uci.edu/dataset/198/steel+plates+faults"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip"
CITATION = ("Buscema, M., Terzi, S., & Tastle, W. (2010). Steel Plates Faults [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5J88N")
LICENSE = "CC BY 4.0 (Creative Commons Attribution 4.0 International)"
SOURCE = "uci_steel_plates_faults"

# The exact files published by UCI. If UCI ever changes them, loading stops
# with a clear error instead of silently using different data.
FILE_HASHES = {
    "Faults.NNA": "08994f5b0185a8e6cc22afe9910fd36834990fb1ca30852ed79012a832fbff55",
    "Faults27x7_var": "887dbfa140b48877b706865ef277550dc0203699243c600e24bfe8b090a9a953",
}
FEATURE_COUNT = 27
LABEL_COUNT = 7

configured = os.environ.get("EVIDENCEOPS_STEEL_DIR")
if configured:
    STEEL_DIR = Path(configured)
else:
    STEEL_DIR = ROOT / "data" / "steel_faults"


def sha256_of(content):
    """The SHA-256 fingerprint of some bytes, as 64 hex characters."""
    return hashlib.sha256(content).hexdigest()


def download_dataset(folder=STEEL_DIR):
    """Download the official zip once, check both files, and save them unchanged."""
    request = Request(DOWNLOAD_URL, headers={"User-Agent": "EvidenceOps dataset download"})
    opener = build_opener(ProxyHandler())
    with opener.open(request, timeout=60) as response:
        archive = response.read(5_000_001)
    if len(archive) > 5_000_000:
        raise ValueError("The downloaded archive is larger than expected")
    saved = []
    with zipfile.ZipFile(io.BytesIO(archive)) as package:
        for name, expected in FILE_HASHES.items():
            # Read only the two expected files; nothing else in the zip is extracted.
            content = package.read(name)
            if sha256_of(content) != expected:
                raise ValueError(f"{name} does not match the published file; refusing to use it")
            folder.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(content)
            saved.append(name)
    return saved


def read_checked_file(folder, name):
    """Read one raw file and stop if it differs from the published version."""
    path = folder / name
    if not path.is_file():
        raise FileNotFoundError("The steel dataset is not downloaded. Run: python main.py steel-download")
    content = path.read_bytes()
    if sha256_of(content) != FILE_HASHES[name]:
        raise ValueError(f"{name} was changed after download; delete it and download again")
    return content


def read_column_names(text):
    """One column name per line. The original file spells one label K_Scatch."""
    names = []
    for line in text.splitlines():
        name = line.strip()
        if name:
            names.append(name)
    if len(names) != FEATURE_COUNT + LABEL_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT + LABEL_COUNT} column names, found {len(names)}")
    return names


def read_rows(text, column_count):
    """Turn tab-separated lines into lists of numbers, checking every value.

    The file uses Windows line endings (\\r\\n), so each line is stripped first.
    """
    rows = []
    line_number = 0
    for line in text.splitlines():
        line_number += 1
        line = line.strip()
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != column_count:
            raise ValueError(f"Line {line_number} has {len(fields)} values, expected {column_count}")
        row = []
        for field in fields:
            value = float(field)  # raises ValueError for text that is not a number
            if not isfinite(value):
                raise ValueError(f"Line {line_number} contains a value that is not a finite number")
            row.append(value)
        rows.append(row)
    return rows


def load_dataset(folder=STEEL_DIR):
    """Load and check the two raw files. Returns {"columns": [...], "rows": [[...], ...]}."""
    names_text = read_checked_file(folder, "Faults27x7_var").decode("ascii")
    data_text = read_checked_file(folder, "Faults.NNA").decode("ascii")
    columns = read_column_names(names_text)
    rows = read_rows(data_text, len(columns))
    return {"columns": columns, "rows": rows, "sha256": FILE_HASHES["Faults.NNA"]}


def fault_of(row, label_names):
    """The name of the one fault label set to 1 in this row."""
    labels = row[FEATURE_COUNT:]
    found = []
    for position in range(len(labels)):
        if labels[position] == 1:
            found.append(label_names[position])
        elif labels[position] != 0:
            raise ValueError("A fault label is neither 0 nor 1")
    if len(found) != 1:
        raise ValueError(f"A record has {len(found)} fault labels; expected exactly 1")
    return found[0]


def percent(part, whole):
    """part as a percentage of whole, rounded to 2 decimals."""
    return round(100 * part / whole, 2)


def summarize(dataset):
    """Calculate every fact the application reports. No model is involved."""
    columns = dataset["columns"]
    rows = dataset["rows"]
    feature_names = columns[:FEATURE_COUNT]
    label_names = columns[FEATURE_COUNT:]
    total = len(rows)
    if total == 0:
        raise ValueError("The dataset has no records")
    # Count records per fault category, in the original column order.
    counts = {}
    for name in label_names:
        counts[name] = 0
    for row in rows:
        counts[fault_of(row, label_names)] += 1
    categories = []
    for name in label_names:
        categories.append({"name": name, "records": counts[name], "percent": percent(counts[name], total)})
    largest = categories[0]
    smallest = categories[0]
    for category in categories:
        if category["records"] > largest["records"]:
            largest = category
        if category["records"] < smallest["records"]:
            smallest = category
    # Simple statistics for every numeric input feature.
    features = []
    for position in range(FEATURE_COUNT):
        values = []
        for row in rows:
            values.append(row[position])
        features.append({"name": feature_names[position], "minimum": min(values), "maximum": max(values),
                         "mean": round(mean(values), 4), "median": median(values)})
    # The two steel-type columns are 0/1 flags; each record should have exactly one.
    a300 = columns.index("TypeOfSteel_A300")
    a400 = columns.index("TypeOfSteel_A400")
    steel_a300 = 0
    steel_a400 = 0
    for row in rows:
        if row[a300] + row[a400] != 1:
            raise ValueError("A record does not have exactly one steel type")
        if row[a300] == 1:
            steel_a300 += 1
        else:
            steel_a400 += 1
    return {"records": total, "columns": len(columns), "feature_names": feature_names,
            "label_names": label_names, "categories": categories, "largest": largest,
            "smallest": smallest, "missing_values": 0, "features": features,
            "steel_a300": steel_a300, "steel_a400": steel_a400, "sha256": dataset["sha256"],
            "source": DATASET_PAGE, "citation": CITATION, "license": LICENSE}


def number_text(value):
    """Show 40.0 as 40 and 0.4782 as 0.4782."""
    if value == int(value):
        return str(int(value))
    return str(value)


def summary_sentences(summary):
    """Write each computed fact as one sentence. These are the only evidence chunks the model sees."""
    total = summary["records"]
    sentences = [
        ("source", f"The Steel Plates Faults dataset (UCI Machine Learning Repository, dataset 198) was "
                   f"created by M. Buscema, S. Terzi and W. Tastle in 2010 and is licensed under "
                   f"{LICENSE}. Source: {DATASET_PAGE}. DOI: https://doi.org/10.24432/C5J88N."),
        ("size", f"The data file Faults.NNA contains {total} records. Each record has {FEATURE_COUNT} "
                 f"numeric input features and {LABEL_COUNT} fault label columns, {summary['columns']} "
                 f"columns in total. No values are missing."),
        ("labels", f"Every record has exactly one of the {LABEL_COUNT} fault labels: "
                   + ", ".join(summary["label_names"]) + "."),
        ("largest", f"The fault category with the most records is {summary['largest']['name']}: "
                    f"{summary['largest']['records']} of {total} records ({summary['largest']['percent']}%). "
                    f"The fault category with the fewest records is {summary['smallest']['name']}: "
                    f"{summary['smallest']['records']} of {total} records ({summary['smallest']['percent']}%)."),
    ]
    for category in summary["categories"]:
        sentences.append(("category " + category["name"],
                          f"Fault category {category['name']}: {category['records']} of {total} records "
                          f"({category['percent']}%)."))
    sentences.append(("features", f"The {FEATURE_COUNT} numeric input features are: "
                      + ", ".join(summary["feature_names"]) + "."))
    sentences.append(("steel types", f"TypeOfSteel_A300 is 1 in {summary['steel_a300']} records and "
                      f"TypeOfSteel_A400 is 1 in {summary['steel_a400']} records. Every record has exactly "
                      f"one of the two steel types."))
    for feature in summary["features"]:
        sentences.append(("feature " + feature["name"],
                          f"Feature {feature['name']} over all {total} records: minimum "
                          f"{number_text(feature['minimum'])}, maximum {number_text(feature['maximum'])}, "
                          f"mean {number_text(feature['mean'])}, median {number_text(feature['median'])}."))
    sentences.append(("limitations", "Every record in this dataset describes a steel plate with a fault. "
                      "The dataset contains no fault-free plates, so it cannot give a fault rate per plate "
                      "produced. It has no dates, production lines, machine settings or causes of faults."))
    sentences.append(("label spelling", "The original variable file spells one fault label K_Scatch; "
                      "the UCI web page also lists it as K_Scratch. This project keeps the original spelling."))
    return sentences


def steel_documents(folder=STEEL_DIR):
    """The computed summary sentences as evidence chunks for search and RAG answers."""
    summary = summarize(load_dataset(folder))
    documents = []
    number = 0
    for topic, text in summary_sentences(summary):
        number += 1
        documents.append({"source": SOURCE, "location": f"summary {number}", "version": summary["sha256"][:12],
                          "topic": topic, "text": text})
    return documents
