import math
import os

import duckdb

FIELD_FOR_SIMILARITY_ENTITY = {
    "COMPANY": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
    "STATE": "Recipient_State",
}

DEFAULT_PROFILE_FIELD = "Nature_of_Payment_or_Transfer_of_Value"

TOP_K = 5
# Entities below this many payments produce spurious 1.000 similarity ties.
MIN_PAYMENTS = 15


def _duckdb_csv_source(csv_path):
    escaped_path = os.path.abspath(csv_path).replace("'", "''")
    return f"read_csv_auto('{escaped_path}', ignore_errors=true)"


def build_profiles(entity_type, csv_path, profile_field=DEFAULT_PROFILE_FIELD):
    entity_field = FIELD_FOR_SIMILARITY_ENTITY[entity_type]
    from_clause = _duckdb_csv_source(csv_path)

    query = f"""
        SELECT
            TRIM("{entity_field}") AS entity_name,
            COALESCE(NULLIF(TRIM("{profile_field}"), ''), 'UNKNOWN') AS profile_value,
            COUNT(*) AS n
        FROM {from_clause}
        WHERE TRIM("{entity_field}") != ''
        GROUP BY entity_name, profile_value
    """
    rows = duckdb.sql(query).fetchall()

    raw_counts = {}
    totals = {}
    for entity_name, profile_value, n in rows:
        raw_counts.setdefault(entity_name, {})[profile_value] = n
        totals[entity_name] = totals.get(entity_name, 0) + n

    profiles = {}
    kept_totals = {}
    for entity_name, counts in raw_counts.items():
        total = totals[entity_name]
        if total < MIN_PAYMENTS:
            continue
        profiles[entity_name] = {k: v / total for k, v in counts.items()}
        kept_totals[entity_name] = total

    return profiles, kept_totals


def cosine_similarity(vec_a, vec_b):
    keys = set(vec_a) | set(vec_b)
    dot = sum(vec_a.get(k, 0.0) * vec_b.get(k, 0.0) for k in keys)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_similar(entity_type, target_entity_value, csv_path, profile_field=DEFAULT_PROFILE_FIELD, top_k=TOP_K):
    profiles, totals = build_profiles(entity_type, csv_path, profile_field)

    target_key = next(
        (name for name in profiles if name.lower() == target_entity_value.lower()),
        None,
    )
    if target_key is None:
        return None

    target_vec = profiles[target_key]

    scored = [
        (name, cosine_similarity(target_vec, vec), totals[name])
        for name, vec in profiles.items()
        if name != target_key
    ]
    scored.sort(key=lambda kv: -kv[1])
    return scored[:top_k]
