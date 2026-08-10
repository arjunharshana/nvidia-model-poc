import argparse
import glob
import os
import re

import duckdb
import joblib
import torch

from similarity import find_similar
from train_cms_qa import IntentEntityModel, tokenize, encode_question

MODEL_PATH = "cms_qa_model.pt"
VOCAB_PATH = "cms_qa_word_vocab.joblib"
INTENT_ENCODER_PATH = "cms_qa_intent_encoder.joblib"
ENTITY_TYPE_ENCODER_PATH = "cms_qa_entity_type_encoder.joblib"
ENTITY_VOCAB_PATH = "data/entity_vocab.joblib"
SAMPLE_CSV = "train/stratified_sample.csv"
DATA_DIR = "data"


def resolve_default_csv():
    full_csvs = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if full_csvs:
        return full_csvs[0]

    print(
        f"[WARNING] No CSV found in {DATA_DIR}/ -- falling back to the "
        f"1000-row training sample ({SAMPLE_CSV}). Answers will be based on "
        f"a small sample, not the full dataset. Pass --csv to point at the "
        f"full CMS file explicitly."
    )
    return SAMPLE_CSV


DEFAULT_CSV = resolve_default_csv()

FIELD_FOR_ENTITY_TYPE = {
    "STATE": "Recipient_State",
    "NATURE": "Nature_of_Payment_or_Transfer_of_Value",
    "CATEGORY": "Product_Category_or_Therapeutic_Area_1",
    "COMPANY": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
    "PROVIDER_TYPE": "Covered_Recipient_Type",
}

GROUP_BY_INTENT_FIELD = {
    "GROUP_SUM_BY_STATE": "Recipient_State",
    "GROUP_SUM_BY_NATURE": "Nature_of_Payment_or_Transfer_of_Value",
    "GROUP_AVG_BY_NATURE": "Nature_of_Payment_or_Transfer_of_Value",
    "GROUP_SUM_BY_CATEGORY": "Product_Category_or_Therapeutic_Area_1",
}

AMOUNT_FIELD = "Total_Amount_of_Payment_USDollars"

# Handled by similarity.find_similar(); value is the expected entity_type.
SIMILARITY_INTENT_ENTITY_TYPE = {
    "SIMILAR_COMPANIES_TO": "COMPANY",
    "SIMILAR_STATES_TO": "STATE",
}


def load_artifacts():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    word2idx = joblib.load(VOCAB_PATH)
    intent_encoder = joblib.load(INTENT_ENCODER_PATH)
    entity_type_encoder = joblib.load(ENTITY_TYPE_ENCODER_PATH)
    entity_vocab = joblib.load(ENTITY_VOCAB_PATH)

    model = IntentEntityModel(
        vocab_size=checkpoint["vocab_size"],
        num_intents=checkpoint["num_intents"],
        num_entity_types=checkpoint["num_entity_types"],
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, word2idx, intent_encoder, entity_type_encoder, entity_vocab


def predict_intent_and_entity_type(question, model, word2idx, intent_encoder, entity_type_encoder):
    ids = encode_question(question, word2idx)
    tokens = torch.tensor(ids, dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        intent_logits, entity_type_logits = model(tokens, offsets)

    intent = intent_encoder.inverse_transform([intent_logits.argmax(dim=1).item()])[0]
    entity_type = entity_type_encoder.inverse_transform([entity_type_logits.argmax(dim=1).item()])[0]
    return intent, entity_type


def extract_entity_value(question, entity_type, entity_vocab):
    if entity_type == "NONE" or entity_type not in entity_vocab:
        return None

    q_lower = question.lower()
    candidates = list(entity_vocab[entity_type])

    # Resolve friendly aliases (e.g. "Physician") back to the CMS string.
    alias_to_canonical = {}
    if entity_type == "PROVIDER_TYPE":
        for canonical, alias in entity_vocab.get("PROVIDER_TYPE_ALIASES", {}).items():
            alias_to_canonical[alias.lower()] = canonical
            candidates.append(alias)

    best_match = None
    best_len = -1
    for candidate in candidates:
        c_lower = candidate.lower()
        pattern = r"(?<![a-z0-9])" + re.escape(c_lower) + r"(?![a-z0-9])"
        if c_lower in alias_to_canonical:
            pattern = r"(?<![a-z0-9])" + re.escape(c_lower) + r"s?(?![a-z0-9])"

        if entity_type == "STATE" and len(candidate) == 2:
            # USPS codes collide with common words, so only match uppercase.
            matched = re.search(r"(?<![A-Za-z0-9])" + re.escape(candidate) + r"(?![A-Za-z0-9])", question)
        else:
            matched = re.search(pattern, q_lower)

        if matched and len(c_lower) > best_len:
            best_match = alias_to_canonical.get(c_lower, candidate)
            best_len = len(c_lower)

    return best_match


def _duckdb_csv_source(csv_path):
    escaped_path = os.path.abspath(csv_path).replace("'", "''")
    return f"read_csv_auto('{escaped_path}', ignore_errors=true)"


def run_aggregation(intent, entity_type, entity_value, csv_path):
    field = FIELD_FOR_ENTITY_TYPE.get(entity_type)
    group_field = GROUP_BY_INTENT_FIELD.get(intent)

    from_clause = _duckdb_csv_source(csv_path)
    amount_expr = f'TRY_CAST("{AMOUNT_FIELD}" AS DOUBLE)'

    where_clause = ""
    params = []
    if field and entity_value:
        where_clause = f'WHERE LOWER(TRIM("{field}")) = LOWER(?)'
        params.append(entity_value)

    if group_field:
        query = f"""
            SELECT COALESCE(NULLIF(TRIM("{group_field}"), ''), 'UNKNOWN') AS group_key,
                   SUM(COALESCE({amount_expr}, 0)) AS total_sum,
                   COUNT(*) AS total_count
            FROM {from_clause}
            {where_clause}
            GROUP BY group_key
        """
        rows = duckdb.sql(query, params=params).fetchall()
        groups = {row[0]: {"sum": row[1], "count": row[2]} for row in rows}
        return {"groups": groups}

    query = f"""
        SELECT SUM(COALESCE({amount_expr}, 0)) AS total_sum,
               COUNT(*) AS total_count
        FROM {from_clause}
        {where_clause}
    """
    row = duckdb.sql(query, params=params).fetchone()
    total = row[0] or 0.0
    count = row[1] or 0
    return {"sum": total, "count": count, "avg": (total / count) if count else 0.0}


def format_answer(intent, entity_type, entity_value, result):
    if "groups" in result:
        lines = [f"Breakdown by {GROUP_BY_INTENT_FIELD[intent]}:"]
        is_avg = intent.startswith("GROUP_AVG")
        for key, stats in sorted(result["groups"].items(), key=lambda kv: -kv[1]["sum"]):
            value = stats["sum"] / stats["count"] if is_avg and stats["count"] else stats["sum"]
            label = "avg" if is_avg else "sum"
            lines.append(f"  {key:40s} {label}=${value:,.2f}  (n={stats['count']})")
        return "\n".join(lines)

    scope = f" for {entity_type.lower()}='{entity_value}'" if entity_value else ""
    if intent.startswith("SUM"):
        return f"Total amount{scope}: ${result['sum']:,.2f}  (n={result['count']})"
    if intent.startswith("COUNT"):
        return f"Payment count{scope}: {result['count']}"
    if intent.startswith("AVG"):
        return f"Average payment amount{scope}: ${result['avg']:,.2f}  (n={result['count']})"
    return str(result)


def format_similarity_answer(entity_type, target_value, results):
    label = entity_type.lower()
    if not results:
        return f"No {label}s with a similar payment-nature profile were found for '{target_value}'."

    lines = [f"Top {len(results)} {label}s most similar to '{target_value}' (by payment-nature distribution):"]
    for name, score, n in results:
        lines.append(f"  {name:50s} similarity={score:.3f}  (n={n})")
    return "\n".join(lines)


def answer(question, csv_path):
    print(f"[Aggregating over: {csv_path}]")

    model, word2idx, intent_encoder, entity_type_encoder, entity_vocab = load_artifacts()

    intent, entity_type = predict_intent_and_entity_type(
        question, model, word2idx, intent_encoder, entity_type_encoder
    )
    entity_value = extract_entity_value(question, entity_type, entity_vocab)

    print(f"Predicted intent:      {intent}")
    print(f"Predicted entity type: {entity_type}")
    print(f"Extracted entity value: {entity_value!r}")
    print()

    if intent in SIMILARITY_INTENT_ENTITY_TYPE:
        expected_entity_type = SIMILARITY_INTENT_ENTITY_TYPE[intent]
        if not entity_value or entity_type != expected_entity_type:
            print(
                f"Could not identify a target {expected_entity_type.lower()} "
                f"in the question -- try naming it explicitly."
            )
            return

        results = find_similar(expected_entity_type, entity_value, csv_path)
        if results is None:
            print(
                f"No payment profile found for {expected_entity_type.lower()}="
                f"{entity_value!r} (not present in the data, or too few payments)."
            )
            return

        print(format_similarity_answer(expected_entity_type, entity_value, results))
        return

    result = run_aggregation(intent, entity_type, entity_value, csv_path)
    print(format_answer(intent, entity_type, entity_value, result))


def main():
    parser = argparse.ArgumentParser(description="Answer questions about CMS Open Payments data.")
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="CSV file to aggregate over (default: first *.csv found in data/, "
        "else falls back to train/stratified_sample.csv with a warning)",
    )
    args = parser.parse_args()

    if args.question:
        answer(args.question, args.csv)
        return

    print("Enter a question (Ctrl+C to exit):")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        answer(question, args.csv)
        print()


if __name__ == "__main__":
    main()
