"""Find the evidence chunks that best match a question. Pure Python: no model is called.

A "chunk" is one paragraph or one CSV record, stored as a dictionary such as
{"source": "sop_v2_en.txt", "location": "paragraph 2", "text": "...", "status": "ACTIVE"}.
Every search function returns at most three chunk dictionaries, best first,
each with a new "score" key.
"""
import re
from collections import Counter
from math import log, sqrt, isfinite

STOP_WORDS = set("a an and are as at be by for from how in is it of on or the to what when where which who why with".split())
ENGLISH_WORD = r"[a-zA-Z][a-zA-Z0-9'-]*"
CHINESE_RUN = r"[㐀-鿿]+"


def current_evidence(documents):
    """Keep active SOPs, current uploads, and CSV records (which have no status)."""
    result = []
    for document in documents:
        status = document.get("status", "").upper()
        if not status or status in ("ACTIVE", "CURRENT", "有效"):
            result.append(document)
    return result


def not_deleted(documents):
    """Keep every version except deleted ones; used when old versions are asked for on purpose."""
    result = []
    for document in documents:
        if document.get("status") != "DELETED":
            result.append(document)
    return result


def english_word_list(text):
    """Lowercase English words without stop words, in order, repeats kept."""
    result = []
    for word in re.findall(ENGLISH_WORD, text.lower()):
        if word not in STOP_WORDS:
            result.append(word)
    return result


def chinese_pair_list(text):
    """Every two neighbouring Chinese characters, repeats kept.

    Chinese has no spaces between words, so "校準紀錄" becomes 校準, 準紀, 紀錄.
    This is a simple stand-in for real word segmentation.
    """
    result = []
    for run in re.findall(CHINESE_RUN, text):
        for index in range(len(run) - 1):
            result.append(run[index:index + 2])
    return result


def words(text):
    """The distinct English words of a text (a set, so each counts once)."""
    return set(english_word_list(text))


def chinese_pairs(text):
    """The distinct Chinese character pairs of a text."""
    return set(chinese_pair_list(text))


def terms(text):
    """English words and Chinese pairs together, with repeats, for BM25 counting."""
    return english_word_list(text) + chinese_pair_list(text)


def is_active(row):
    """True for an ACTIVE English SOP or a 有效 (valid) Chinese SOP."""
    return row.get("status", "").upper() in ("ACTIVE", "有效")


def by_score(row):
    """Sort key: the score alone."""
    return row["score"]


def by_score_then_active(row):
    """Sort key: higher score first; on a tie, active documents first."""
    return (row["score"], is_active(row))


def search_documents(documents, question):
    """The original baseline: score = number of shared English words."""
    question_words = words(question)
    results = []
    for document in documents:
        matched = question_words & words(document["text"])  # & keeps words found in both sets
        if matched:
            result = document.copy()
            result["score"] = len(matched)
            result["matched"] = sorted(matched)
            results.append(result)
    # Python's sort is stable: chunks with equal scores keep their file order.
    results.sort(key=by_score, reverse=True)
    return results[:3]


def search_bilingual(documents, question):
    """Score = shared English words + shared Chinese character pairs."""
    question_words = words(question)
    question_pairs = chinese_pairs(question)
    results = []
    for document in documents:
        matched_words = question_words & words(document["text"])
        matched_pairs = question_pairs & chinese_pairs(document["text"])
        score = len(matched_words) + len(matched_pairs)
        if score > 0:
            result = document.copy()
            result["score"] = score
            result["matched"] = sorted(matched_words | matched_pairs)
            result["method"] = "keyword"
            results.append(result)
    results.sort(key=by_score_then_active, reverse=True)
    return results[:3]


def bm25_term_score(frequency, chunks_with_term, chunk_count, chunk_length, average_length):
    """BM25 score for one question term in one chunk.

    rarity: a term found in few chunks is worth more than a common one.
    saturation: the 2nd, 3rd... repeat of a term adds less and less (k1 = 1.2).
    length: a match in a short chunk counts more than in a long one (b = 0.75).
    """
    rarity = log(1 + (chunk_count - chunks_with_term + 0.5) / (chunks_with_term + 0.5))
    length_factor = 0.25 + 0.75 * chunk_length / average_length
    return rarity * frequency * 2.2 / (frequency + 1.2 * length_factor)


def search_bm25(documents, question):
    """Rank chunks with BM25: term frequency, term rarity and chunk length."""
    question_terms = sorted(set(terms(question)))
    if not question_terms or not documents:
        return []
    # Count every term in every chunk once, up front.
    chunk_counts = []
    total_length = 0
    for document in documents:
        counts = Counter(terms(document["text"]))
        chunk_counts.append(counts)
        total_length += sum(counts.values())
    average_length = total_length / len(documents)
    if average_length == 0:
        return []
    # How many chunks contain each question term?
    chunks_with = {}
    for term in question_terms:
        chunks_with[term] = 0
        for counts in chunk_counts:
            if term in counts:
                chunks_with[term] += 1
    results = []
    for index, document in enumerate(documents):
        counts = chunk_counts[index]
        chunk_length = sum(counts.values())
        score = 0.0
        matched = []
        for term in question_terms:
            frequency = counts[term]
            if frequency == 0:
                continue
            matched.append(term)
            score += bm25_term_score(frequency, chunks_with[term], len(documents), chunk_length, average_length)
        if score:
            result = document.copy()
            result["score"] = round(score, 6)
            result["matched"] = matched
            result["method"] = "bm25"
            results.append(result)
    results.sort(key=by_score_then_active, reverse=True)
    return results[:3]


def is_valid_number(value):
    """True for a real, finite int or float. True/False do not count as numbers here."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return isfinite(value)


def vector_length(vector):
    """The length of a vector: the square root of the sum of its squared values."""
    total = 0
    for value in vector:
        total += value * value
    return sqrt(total)


def search_vectors(documents, query_vector, document_vectors):
    """Rank chunks by cosine similarity between their vectors and the question's vector.

    cosine similarity = dot product / (length of a * length of b), from -1 to 1.
    This function only ranks numbers it is given; it never calls a model.
    """
    if not isinstance(documents, (list, tuple)) or not isinstance(document_vectors, (list, tuple)):
        raise ValueError("Documents and document vectors must be lists")
    if len(documents) != len(document_vectors):
        raise ValueError("Each document needs exactly one vector")
    if not isinstance(query_vector, (list, tuple)) or not 1 <= len(query_vector) <= 4096:
        raise ValueError("Query vector must have 1 to 4096 numbers")
    for value in query_vector:
        if not is_valid_number(value):
            raise ValueError("Query vector contains an invalid number")
    query_length = vector_length(query_vector)
    if query_length == 0 or not isfinite(query_length):
        raise ValueError("Query vector has an invalid length")
    results = []
    for index, document in enumerate(documents):
        vector = document_vectors[index]
        if not isinstance(vector, (list, tuple)) or len(vector) != len(query_vector):
            raise ValueError("Document vector dimensions do not match")
        for value in vector:
            if not is_valid_number(value):
                raise ValueError("Document vector contains an invalid number")
        length = vector_length(vector)
        if length == 0 or not isfinite(length):
            raise ValueError("Document vector has an invalid length")
        dot_product = 0
        for position in range(len(vector)):
            dot_product += query_vector[position] * vector[position]
        if not isfinite(dot_product):
            raise ValueError("Vector similarity is not finite")
        result = document.copy()
        result["score"] = round(dot_product / (query_length * length), 6)
        result["method"] = "semantic"
        results.append(result)
    results.sort(key=by_score, reverse=True)
    return results[:3]


def chunk_key(row):
    """One chunk's identity: source file, version and location."""
    return (row.get("source"), row.get("version"), row.get("location"))


def by_fusion_score(row):
    """Sort key for hybrid search."""
    return row["fusion_score"]


def hybrid_search(documents, question, semantic_search=None):
    """Combine keyword and vector rankings with Reciprocal Rank Fusion (RRF).

    Each chunk earns 1 / (60 + rank) from every ranking it appears in. RRF uses
    only the rank, so keyword scores and cosine scores never need the same scale.
    semantic_search is a function that takes (documents, question).
    """
    if not callable(semantic_search):
        raise RuntimeError("No embedding search is configured")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A question is required")
    # Remember every supplied chunk, so semantic_search cannot add outside text.
    known_text = {}
    for document in documents:
        known_text[chunk_key(document)] = document["text"]
    semantic_results = semantic_search(documents, question)
    if not isinstance(semantic_results, list):
        raise ValueError("Semantic search must return a list of evidence dictionaries")
    for result in semantic_results:
        if not isinstance(result, dict):
            raise ValueError("Semantic search returned an invalid evidence item")
        key = chunk_key(result)
        if key not in known_text or result.get("text") != known_text[key]:
            raise ValueError("Semantic search returned evidence outside the supplied documents")
    rankings = [("keyword", search_bilingual(documents, question)), ("semantic", semantic_results)]
    fused = {}
    for method, results in rankings:
        seen = set()
        rank = 0
        for result in results:
            rank += 1
            key = chunk_key(result)
            if key in seen:
                continue
            seen.add(key)
            if key not in fused:
                row = result.copy()
                row["fusion_score"] = 0.0
                row["methods"] = []
                fused[key] = row
            fused[key]["fusion_score"] += 1 / (60 + rank)
            fused[key]["methods"].append(method)
    combined = list(fused.values())
    combined.sort(key=by_fusion_score, reverse=True)
    return combined[:3]
