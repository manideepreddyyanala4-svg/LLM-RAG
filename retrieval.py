"""The original English keyword baseline, with an interchangeable search function."""
import re
from collections import Counter
from math import log, sqrt, isfinite

STOP_WORDS = set("a an and are as at be by for from how in is it of on or the to what when where which who why with".split())


def current_evidence(documents):
    """Keep active SOPs, current uploads, and CSV records without a status."""
    result = []
    for document in documents:
        status = document.get("status", "").upper()
        if not status or status in ("ACTIVE", "CURRENT", "有效"):
            result.append(document)
    return result


def words(text):
    """Text in; distinct lowercase English tokens out. Chinese is not segmented."""
    found = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*", text.lower())
    return set(found) - STOP_WORDS


def search_documents(documents, question):
    """Return up to three evidence dictionaries with positive overlap scores."""
    query = words(question)
    results = []
    for document in documents:
        matched = query & words(document["text"])
        if matched:
            result = document.copy()
            result["score"] = len(matched)
            result["matched"] = sorted(matched)
            results.append(result)
    # Stable sorting preserves source/paragraph order when scores tie.
    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:3]


def search_bilingual(documents, question):
    """Search English tokens plus overlapping Chinese character pairs."""
    english = words(question)
    chinese = chinese_pairs(question)
    results = []
    for document in documents:
        matched_en = english & words(document["text"])
        matched_zh = chinese & chinese_pairs(document["text"])
        score = len(matched_en) + len(matched_zh)
        if score:
            result = document.copy()
            result.update(score=score, matched=sorted(matched_en | matched_zh), method="keyword")
            results.append(result)
    results.sort(key=lambda row: (row["score"], row.get("status", "").upper() in ("ACTIVE", "有效")), reverse=True)
    return results[:3]


def chinese_pairs(text):
    """Return adjacent Han character pairs; this is not word segmentation."""
    pairs = set()
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        for index in range(len(run) - 1):
            pairs.add(run[index:index + 2])
    return pairs


def terms(text):
    """Return repeated English words and Han pairs for frequency scoring."""
    found = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*", text.lower())
    result = [word for word in found if word not in STOP_WORDS]
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        for index in range(len(run) - 1):
            result.append(run[index:index + 2])
    return result


def search_bm25(documents, question):
    """Rank chunks by term frequency, rarity, and document length."""
    query = set(terms(question))
    if not query or not documents:
        return []
    tokenized = []
    total_length = 0
    for document in documents:
        counts = Counter(terms(document["text"]))
        tokenized.append(counts)
        total_length += sum(counts.values())
    average_length = total_length / len(documents)
    if average_length == 0:
        return []
    document_frequency = {}
    for word in query:
        document_frequency[word] = 0
        for counts in tokenized:
            if word in counts:
                document_frequency[word] += 1
    results = []
    for index, document in enumerate(documents):
        counts = tokenized[index]
        length = sum(counts.values())
        score = 0.0
        matched = []
        for word in query:
            frequency = counts[word]
            if not frequency:
                continue
            matched.append(word)
            containing = document_frequency[word]
            rarity = log(1 + (len(documents) - containing + 0.5) / (containing + 0.5))
            score += rarity * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average_length))
        if score:
            result = document.copy()
            result.update(score=round(score, 6), matched=sorted(matched), method="bm25")
            results.append(result)
    results.sort(key=lambda row: (row["score"], row.get("status", "").upper() in ("ACTIVE", "有效")), reverse=True)
    return results[:3]


def search_vectors(documents, query_vector, document_vectors):
    """Rank supplied evidence vectors by cosine similarity; no model is called."""
    if not isinstance(documents, (list, tuple)) or not isinstance(document_vectors, (list, tuple)):
        raise ValueError("Documents and document vectors must be lists")
    if len(documents) != len(document_vectors):
        raise ValueError("Each document needs exactly one vector")
    if not isinstance(query_vector, (list, tuple)) or not 1 <= len(query_vector) <= 4096:
        raise ValueError("Query vector must have 1 to 4096 numbers")
    for value in query_vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError("Query vector contains an invalid number")
    query_length = sqrt(sum(value * value for value in query_vector))
    if query_length == 0 or not isfinite(query_length):
        raise ValueError("Query vector has an invalid length")
    results = []
    for index, document in enumerate(documents):
        vector = document_vectors[index]
        if not isinstance(vector, (list, tuple)) or len(vector) != len(query_vector):
            raise ValueError("Document vector dimensions do not match")
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError("Document vector contains an invalid number")
        length = sqrt(sum(value * value for value in vector))
        if length == 0 or not isfinite(length):
            raise ValueError("Document vector has an invalid length")
        dot_product = 0
        for position, value in enumerate(vector):
            dot_product += query_vector[position] * value
        if not isfinite(dot_product):
            raise ValueError("Vector similarity is not finite")
        result = document.copy()
        result.update(score=round(dot_product / (query_length * length), 6), method="semantic")
        results.append(result)
    results.sort(key=lambda row: row["score"], reverse=True)
    return results[:3]


def hybrid_search(documents, question, semantic_search=None):
    """Fuse lexical and supplied semantic ranks for the same evidence chunks."""
    if not callable(semantic_search):
        raise RuntimeError("No embedding search is configured")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A question is required")
    known = {}
    for document in documents:
        key = (document["source"], document.get("version"), document["location"])
        known[key] = document["text"]
    semantic_results = semantic_search(documents, question)
    if not isinstance(semantic_results, list):
        raise ValueError("Semantic search must return a list of evidence dictionaries")
    for result in semantic_results:
        if not isinstance(result, dict):
            raise ValueError("Semantic search returned an invalid evidence item")
        key = (result.get("source"), result.get("version"), result.get("location"))
        if key not in known or result.get("text") != known[key]:
            raise ValueError("Semantic search returned evidence outside the supplied documents")
    ranked = {}
    for method, results in (("keyword", search_bilingual(documents, question)),
                            ("semantic", semantic_results)):
        seen = set()
        for position, result in enumerate(results, 1):
            key = (result["source"], result.get("version"), result["location"])
            if key in seen:
                continue
            seen.add(key)
            if key not in ranked:
                ranked[key] = dict(result, fusion_score=0.0, methods=[])
            ranked[key]["fusion_score"] += 1 / (60 + position)
            ranked[key]["methods"].append(method)
    return sorted(ranked.values(), key=lambda row: row["fusion_score"], reverse=True)[:3]
