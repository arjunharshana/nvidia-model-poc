import hashlib
import os

import joblib
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
ENTITY_VOCAB_PATH = "data/entity_vocab.joblib"
EMBEDDINGS_CACHE_PATH = "entity_embeddings.joblib"

# Cosine similarity threshold for COMPANY/CATEGORY/NATURE/PROVIDER_TYPE; STATE bypasses this entirely.
SIMILARITY_THRESHOLD = 0.65

MAX_SPAN_TOKENS = 6

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "by", "to", "of", "for", "and", "or",
    "is", "are", "was", "were", "did", "do", "does", "with", "from", "as",
    "it", "this", "that", "what", "which", "who", "how", "many", "much",
}

ENTITY_TYPES = ["STATE", "NATURE", "CATEGORY", "COMPANY", "PROVIDER_TYPE"]

# STATE is matched via this closed-set dictionary, not embeddings -- see _match_state.
STATE_ALIASES = {
    "alabama": "AL", "al": "AL",
    "alaska": "AK", "ak": "AK",
    "arizona": "AZ", "az": "AZ",
    "arkansas": "AR", "ar": "AR",
    "california": "CA", "calif": "CA", "cali": "CA", "ca": "CA",
    "colorado": "CO", "colo": "CO", "co": "CO",
    "connecticut": "CT", "conn": "CT", "ct": "CT",
    "delaware": "DE", "del": "DE", "de": "DE",
    "district of columbia": "DC", "washington dc": "DC", "washington d.c.": "DC", "dc": "DC",
    "florida": "FL", "fla": "FL", "fl": "FL",
    "georgia": "GA", "ga": "GA",
    "hawaii": "HI", "hi": "HI",
    "idaho": "ID", "id": "ID",
    "illinois": "IL", "ill": "IL", "il": "IL",
    "indiana": "IN", "ind": "IN", "in": "IN",
    "iowa": "IA", "ia": "IA",
    "kansas": "KS", "kan": "KS", "kans": "KS", "ks": "KS",
    "kentucky": "KY", "ky": "KY",
    "louisiana": "LA", "la": "LA",
    "maine": "ME", "me": "ME",
    "maryland": "MD", "md": "MD",
    "massachusetts": "MA", "mass": "MA", "ma": "MA",
    "michigan": "MI", "mich": "MI", "mi": "MI",
    "minnesota": "MN", "minn": "MN", "mn": "MN",
    "mississippi": "MS", "miss": "MS", "ms": "MS",
    "missouri": "MO", "mo": "MO",
    "montana": "MT", "mont": "MT", "mt": "MT",
    "nebraska": "NE", "neb": "NE", "nebr": "NE", "ne": "NE",
    "nevada": "NV", "nev": "NV", "nv": "NV",
    "new hampshire": "NH", "nh": "NH",
    "new jersey": "NJ", "nj": "NJ",
    "new mexico": "NM", "n mex": "NM", "nm": "NM",
    "new york": "NY", "ny": "NY",
    "north carolina": "NC", "n carolina": "NC", "nc": "NC",
    "north dakota": "ND", "n dakota": "ND", "nd": "ND",
    "ohio": "OH", "oh": "OH",
    "oklahoma": "OK", "okla": "OK", "ok": "OK",
    "oregon": "OR", "ore": "OR", "or": "OR",
    "pennsylvania": "PA", "penn": "PA", "pa": "PA",
    "puerto rico": "PR", "pr": "PR",
    "rhode island": "RI", "ri": "RI",
    "south carolina": "SC", "s carolina": "SC", "sc": "SC",
    "south dakota": "SD", "s dakota": "SD", "sd": "SD",
    "tennessee": "TN", "tenn": "TN", "tn": "TN",
    "texas": "TX", "tex": "TX", "tx": "TX",
    "utah": "UT", "ut": "UT",
    "vermont": "VT", "vt": "VT",
    "virginia": "VA", "va": "VA",
    "washington": "WA", "wash": "WA", "wa": "WA",
    "west virginia": "WV", "w virginia": "WV", "wv": "WV",
    "wisconsin": "WI", "wis": "WI", "wisc": "WI", "wi": "WI",
    "wyoming": "WY", "wy": "WY",
    "armed forces pacific": "AP", "ap": "AP",
}

# Short codes require an exact uppercase token match (lowercase collides with common words).
_AMBIGUOUS_STATE_CODES = {
    alias for alias in STATE_ALIASES if len(alias) <= 3
}

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _vocab_fingerprint(vocab_path):
    with open(vocab_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _build_embeddings(entity_vocab):
    model = _get_model()
    labels_by_type = {}
    embeddings_by_type = {}

    for entity_type in ENTITY_TYPES:
        if entity_type == "STATE":
            continue

        values = list(entity_vocab.get(entity_type, []))

        alias_to_canonical = {}
        if entity_type == "PROVIDER_TYPE":
            for canonical, alias in entity_vocab.get("PROVIDER_TYPE_ALIASES", {}).items():
                alias_to_canonical[alias] = canonical
                values.append(alias)

        if not values:
            labels_by_type[entity_type] = []
            embeddings_by_type[entity_type] = np.zeros((0, 384), dtype=np.float32)
            continue

        vectors = model.encode(values, normalize_embeddings=True, show_progress_bar=False)
        labels_by_type[entity_type] = [alias_to_canonical.get(v, v) for v in values]
        embeddings_by_type[entity_type] = np.asarray(vectors, dtype=np.float32)

    return labels_by_type, embeddings_by_type


def load_or_build_cache(vocab_path=ENTITY_VOCAB_PATH, cache_path=EMBEDDINGS_CACHE_PATH):
    entity_vocab = joblib.load(vocab_path)
    fingerprint = _vocab_fingerprint(vocab_path)

    if os.path.exists(cache_path):
        cached = joblib.load(cache_path)
        if (
            cached.get("vocab_fingerprint") == fingerprint
            and cached.get("model_name") == MODEL_NAME
            and cached.get("format") == "v2"
        ):
            return cached["labels_by_type"], cached["embeddings_by_type"]

    labels_by_type, embeddings_by_type = _build_embeddings(entity_vocab)
    joblib.dump(
        {
            "format": "v2",
            "vocab_fingerprint": fingerprint,
            "model_name": MODEL_NAME,
            "labels_by_type": labels_by_type,
            "embeddings_by_type": embeddings_by_type,
        },
        cache_path,
    )
    return labels_by_type, embeddings_by_type


def _candidate_spans(question, max_tokens=MAX_SPAN_TOKENS):
    # Longest spans first, so a full multi-word vocab entry wins over a shorter sub-span.
    words = question.split()
    n = len(words)
    spans = []
    for length in range(min(max_tokens, n), 0, -1):
        for start in range(0, n - length + 1):
            span_words = words[start:start + length]
            if length == 1 and span_words[0].lower() in STOPWORDS:
                continue
            spans.append(" ".join(span_words))
    return spans


_PUNCT = ".,?!;:'\""


def _strip_span_punct(span):
    return " ".join(w.strip(_PUNCT) for w in span.split())


def _match_state(question):
    spans = _candidate_spans(question)
    raw_tokens_upper = {t.strip(_PUNCT) for t in question.split()}

    for span in spans:
        key = _strip_span_punct(span).lower()
        code = STATE_ALIASES.get(key)
        if code is None:
            continue
        if key in _AMBIGUOUS_STATE_CODES:
            if code in raw_tokens_upper:
                return code
            continue
        return code
    return None


def _embedding_best_match(question, labels, embeddings, threshold):
    spans = _candidate_spans(question)
    if not spans or embeddings is None or embeddings.shape[0] == 0:
        return None

    model = _get_model()
    span_vectors = model.encode(spans, normalize_embeddings=True, show_progress_bar=False)
    span_vectors = np.asarray(span_vectors, dtype=np.float32)

    sims = span_vectors @ embeddings.T  # (num_spans, num_labels), spans ordered longest-first

    for span_idx in range(sims.shape[0]):
        label_idx = int(np.argmax(sims[span_idx]))
        score = float(sims[span_idx, label_idx])
        if score >= threshold:
            return labels[label_idx], score
    return None


def best_match(question, entity_type, labels_by_type, embeddings_by_type, threshold=SIMILARITY_THRESHOLD):
    if entity_type == "STATE":
        code = _match_state(question)
        return (code, 1.0) if code else None

    labels = labels_by_type.get(entity_type, [])
    embeddings = embeddings_by_type.get(entity_type)
    if not labels:
        return None
    return _embedding_best_match(question, labels, embeddings, threshold)


def top_matches(question, entity_type, labels_by_type, embeddings_by_type, threshold=SIMILARITY_THRESHOLD, top_k=2):
    if entity_type == "STATE":
        raw_tokens_upper = {t.strip(_PUNCT) for t in question.split()}
        spans = _candidate_spans(question)
        results = []
        seen = set()
        for span in spans:
            key = _strip_span_punct(span).lower()
            code = STATE_ALIASES.get(key)
            if code is None or code in seen:
                continue
            if key in _AMBIGUOUS_STATE_CODES and code not in raw_tokens_upper:
                continue
            seen.add(code)
            results.append(code)
            if len(results) >= top_k:
                break
        return results

    labels = labels_by_type.get(entity_type, [])
    embeddings = embeddings_by_type.get(entity_type)
    if not labels or embeddings is None or embeddings.shape[0] == 0:
        return []

    spans = _candidate_spans(question)
    if not spans:
        return []

    model = _get_model()
    span_vectors = model.encode(spans, normalize_embeddings=True, show_progress_bar=False)
    span_vectors = np.asarray(span_vectors, dtype=np.float32)

    sims = span_vectors @ embeddings.T  # (num_spans, num_labels), spans ordered longest-first

    results = []
    seen = set()
    for span_idx in range(sims.shape[0]):
        if len(results) >= top_k:
            break
        label_idx = int(np.argmax(sims[span_idx]))
        score = float(sims[span_idx, label_idx])
        if score < threshold:
            continue
        label = labels[label_idx]
        if label not in seen:
            seen.add(label)
            results.append(label)
    return results
