
import csv
import json
import random
import sys
from pathlib import Path

import joblib

DEFAULT_CSV = "train/stratified_sample.csv"
DEFAULT_TARGET = 7500
OUT_JSONL = "train/qa_dataset.jsonl"
VOCAB_OUT = "data/entity_vocab.joblib"


OVERALL_PER_INTENT = 450

SUM_TOTAL_TEMPLATES = [
    "What is the sum of all payments?",
    "What is the total dollar amount of all payments?",
    "Calculate the total dollar amount paid across all records.",
    "What is the sum of every payment in the dataset?",
    "Give me the grand total of all payments.",
    "What is the grand total dollar amount paid?",
    "Add up all the payment amounts.",
    "Sum all the payments in the dataset.",
    "What is the combined dollar amount of all payments?",
    "Compute the sum total of every payment amount.",
    "What's the aggregate dollar amount of all payments?",
    "Total up all the payment amounts in dollars.",
    "Give me the sum total of payment amounts.",
    "What is the total dollar value of all payments combined?",
    "How much money was paid out in total dollars?",
    "How much money changed hands in total?",
    "Tell me the overall dollar amount paid.",
    "What's the total spend across every payment?",
    "I need the combined total of every payment made.",
    "Everything added up, how much was paid?",
]

COUNT_TOTAL_TEMPLATES = [
    "How many payment records are there in total?",
    "Count all the payments in the dataset.",
    "How many total records are there?",
    "How many payments are there overall?",
    "What is the total number of payments in the dataset?",
    "How many payment records exist?",
    "How many total payment records exist?",
    "How many payments are in the dataset?",
    "Count the number of payment records.",
    "What is the count of all payments?",
    "How many rows are in the dataset?",
    "Give me a count of all payment records.",
    "How many individual payments were made?",
    "What's the number of payments recorded?",
    "How many entries are in the payments dataset?",
    "Count how many payments exist.",
    "Tell me the total number of transactions logged.",
    "What's the payment count across the board?",
    "I want to know how many payments happened in total.",
    "Just how many payment entries are we talking about?",
]

AVG_TOTAL_TEMPLATES = [
    "What is the average payment amount overall?",
    "What's the average amount paid per payment?",
    "On average, how much is each payment worth?",
    "What is the mean payment amount across all records?",
    "What is the average dollar value of a payment?",
    "What's the typical payment amount?",
    "On average, how much does each payment amount to?",
    "What is the mean amount per payment?",
    "Calculate the average payment value.",
    "What's the average dollar amount per transaction?",
    "What is the average size of a payment?",
    "On average, what does a payment amount to?",
    "What's a typical payment worth?",
    "Roughly how much does each payment come out to on average?",
    "What's the mean transaction size across everything?",
]

STATE_FIELD = "Recipient_State"
NATURE_FIELD = "Nature_of_Payment_or_Transfer_of_Value"
CATEGORY_FIELD = "Product_Category_or_Therapeutic_Area_1"
COMPANY_FIELD = "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name"
PROVIDER_TYPE_FIELD = "Covered_Recipient_Type"

ENTITY_TYPES = ["STATE", "NATURE", "CATEGORY", "COMPANY", "PROVIDER_TYPE"]

# Friendly names for question generation; entity_value stays the formal CMS string.
PROVIDER_TYPE_ALIASES = {
    "Covered Recipient Physician": "Physician",
    "Covered Recipient Non-Physician Practitioner": "Non-Physician Practitioner",
    "Covered Recipient Teaching Hospital": "Teaching Hospital",
}


def normalize_dedupe(values):
    """Case-insensitive dedupe, keeping the most frequent original casing."""
    counts = {}
    for v in values:
        v = (v or "").strip()
        if not v:
            continue
        key = v.lower()
        counts.setdefault(key, {}).setdefault(v, 0)
        counts[key][v] += 1

    canonical = []
    for key, variants in counts.items():
        best = max(variants.items(), key=lambda kv: kv[1])[0]
        canonical.append(best)
    return canonical


def extract_vocab(csv_path):
    states, natures, categories, companies, provider_types = [], [], [], [], []

    with open(csv_path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            states.append(row.get(STATE_FIELD))
            natures.append(row.get(NATURE_FIELD))
            categories.append(row.get(CATEGORY_FIELD))
            companies.append(row.get(COMPANY_FIELD))
            provider_types.append(row.get(PROVIDER_TYPE_FIELD))

    return {
        "STATE": sorted(normalize_dedupe(states)),
        "NATURE": sorted(normalize_dedupe(natures)),
        "CATEGORY": sorted(normalize_dedupe(categories)),
        "COMPANY": sorted(normalize_dedupe(companies)),
        "PROVIDER_TYPE": sorted(normalize_dedupe(provider_types)),
    }


def sample(values, n, rng):
    pool = list(values)
    rng.shuffle(pool)
    return pool[:n]


def build_overall_examples(rng, per_intent=OVERALL_PER_INTENT):
    """Fixed per_intent examples each for SUM_TOTAL/COUNT_TOTAL/AVG_TOTAL."""
    examples = []
    for templates, intent in [
        (SUM_TOTAL_TEMPLATES, "SUM_TOTAL"),
        (COUNT_TOTAL_TEMPLATES, "COUNT_TOTAL"),
        (AVG_TOTAL_TEMPLATES, "AVG_TOTAL"),
    ]:
        for _ in range(per_intent):
            examples.append(
                {
                    "question": rng.choice(templates),
                    "intent": intent,
                    "entity_type": "NONE",
                    "entity_value": "",
                }
            )
    return examples


def build_dataset(vocab, target_size, rng):
    examples = []

    def add(question, intent, entity_type="NONE", entity_value=""):
        examples.append(
            {
                "question": question,
                "intent": intent,
                "entity_type": entity_type,
                "entity_value": entity_value,
            }
        )

    def add_multi(question, intent, entities):
        examples.append(
            {
                "question": question,
                "intent": intent,
                "entities": [
                    {"entity_type": t, "entity_value": v} for t, v in entities
                ],
            }
        )

    state_sample = sample(vocab["STATE"], min(50, len(vocab["STATE"])), rng)
    nature_sample = sample(vocab["NATURE"], min(16, len(vocab["NATURE"])), rng)
    category_sample = sample(vocab["CATEGORY"], min(60, len(vocab["CATEGORY"])), rng)
    company_sample = sample(vocab["COMPANY"], min(80, len(vocab["COMPANY"])), rng)
    provider_type_sample = sample(
        vocab["PROVIDER_TYPE"], min(10, len(vocab["PROVIDER_TYPE"])), rng
    )

    entity_templates = {
        "STATE": {
            "SUM_BY_STATE": [
                "What is the total paid in {v}?",
                "Total amount of payments in {v}.",
                "How much money went to recipients in {v}?",
                "Sum the payments made in {v}.",
                "What's the overall dollar amount paid out in {v}?",
                "Add up every payment made to {v}.",
                "I'd like the combined total for {v}.",
                "{v} recipients received how much in total?",
            ],
            "COUNT_BY_STATE": [
                "How many payments were made in {v}?",
                "Count the number of payments in {v}.",
                "How many payment records are there for {v}?",
                "What's the number of payments in {v}?",
                "Tell me the count of payments made in {v}.",
                "How many transactions happened in {v}?",
                "{v} had how many payments made to it?",
            ],
            "AVG_BY_STATE": [
                "What is the average payment amount in {v}?",
                "Average payment amount for recipients in {v}.",
                "What's the typical payment size in {v}?",
                "On average, how much is a payment in {v} worth?",
                "What's the mean payment amount for {v}?",
            ],
            "SIMILAR_STATES_TO": [
                "Which state has similar payment types to {v}?",
                "Which states are most similar to {v} in terms of payments?",
                "Find states with a similar payment pattern to {v}.",
                "What state is most like {v} based on payment nature?",
                "Which other states resemble {v}'s payment breakdown?",
                "Show me states similar to {v}.",
                "What states behave like {v} payment-wise?",
            ],
        },
        "NATURE": {
            "SUM_BY_NATURE": [
                "What is the total amount paid for {v}?",
                "Total payments categorized as {v}.",
                "Sum of all {v} payments.",
                "How much money went toward {v}?",
                "What's the combined dollar amount for {v} payments?",
                "Add up all the {v} payments.",
            ],
            "COUNT_BY_NATURE": [
                "How many payments are there for {v}?",
                "Count payments of nature {v}.",
                "How many {v} payments were made?",
                "What's the number of {v} payments?",
                "Tell me how many times {v} was the payment type.",
            ],
            "AVG_BY_NATURE": [
                "What is the average payment amount for {v}?",
                "Average amount for {v} payments.",
                "What's the typical size of a {v} payment?",
                "On average, how much is a {v} payment worth?",
            ],
        },
        "CATEGORY": {
            "SUM_BY_CATEGORY": [
                "What is the total amount paid for {v}?",
                "Total payments for the {v} category.",
                "Sum of payments related to {v}.",
                "How much was spent on {v}?",
                "What's the combined total for the {v} category?",
            ],
            "COUNT_BY_CATEGORY": [
                "How many payments are there for {v}?",
                "How many payments were for {v}?",
                "Count the payments in the {v} category.",
                "What's the number of payments tied to {v}?",
                "Tell me the payment count for {v}.",
            ],
            "AVG_BY_CATEGORY": [
                "What is the average payment for {v}?",
                "Average payment amount for {v} related payments.",
                "What's the typical payment size for {v}?",
                "On average, how much is a {v} payment?",
            ],
        },
        "COMPANY": {
            "SUM_BY_COMPANY": [
                "How much did {v} pay in total?",
                "What is the total amount paid by {v}?",
                "Sum of payments made by {v}.",
                "What's the overall amount {v} paid out?",
                "Add up everything {v} has paid.",
                "{v} paid out how much in total?",
            ],
            "COUNT_BY_COMPANY": [
                "How many payments did {v} make?",
                "Count the payments made by {v}.",
                "What's the number of payments from {v}?",
                "How many times did {v} make a payment?",
            ],
            "AVG_BY_COMPANY": [
                "What is the average payment made by {v}?",
                "Average amount per payment from {v}.",
                "What's the typical payment size from {v}?",
                "On average, how much does {v} pay per transaction?",
            ],
            "SIMILAR_COMPANIES_TO": [
                "Which company has a similar payment pattern to {v}?",
                "What companies are most similar to {v}?",
                "Find companies with payment behavior like {v}.",
                "Which manufacturer resembles {v} in terms of payment types?",
                "Show me companies similar to {v}.",
                "What other companies pay similarly to {v}?",
                "Who else pays like {v} does?",
            ],
        },
        "PROVIDER_TYPE": {
            "SUM_BY_PROVIDER_TYPE": [
                "What is the total amount paid to {v}s?",
                "Total payments made to recipients of type {v}.",
                "How much did {v}s get paid in total?",
                "How much money did {v}s receive overall?",
                "Sum of all payments made to {v}s.",
                "What's the combined amount paid out to {v}s?",
            ],
            "COUNT_BY_PROVIDER_TYPE": [
                "How many payments went to {v}s?",
                "Count payments made to {v} recipients.",
                "How many {v}s received payments?",
                "How many payments were made to {v} recipients?",
                "What's the number of payments to {v}s?",
            ],
        },
    }

    samples_by_type = {
        "STATE": state_sample,
        "NATURE": nature_sample,
        "CATEGORY": category_sample,
        "COMPANY": company_sample,
        "PROVIDER_TYPE": provider_type_sample,
    }

    for entity_type, intents in entity_templates.items():
        for value in samples_by_type[entity_type]:
            fill_value = PROVIDER_TYPE_ALIASES.get(value, value) if entity_type == "PROVIDER_TYPE" else value
            for intent, templates in intents.items():
                for template in templates:
                    add(template.format(v=fill_value), intent, entity_type, value)

    # Two-filter combinations: smaller per-pair samples than the single-entity
    # block above, since the number of (value_a, value_b) pairs grows fast.
    # NATURE only has ~16 distinct values total, so cover all of them -- a
    # partial sample would silently drop real natures (e.g. "Consulting Fee")
    # from every two-filter combo that uses it.
    two_filter_state_sample = sample(vocab["STATE"], min(30, len(vocab["STATE"])), rng)
    two_filter_nature_sample = list(vocab["NATURE"])
    two_filter_category_sample = sample(vocab["CATEGORY"], min(25, len(vocab["CATEGORY"])), rng)
    two_filter_company_sample = sample(vocab["COMPANY"], min(30, len(vocab["COMPANY"])), rng)

    two_filter_combos = [
        # (entity_type_a, sample_a, entity_type_b, sample_b, intent, templates)
        (
            "STATE", two_filter_state_sample, "NATURE", two_filter_nature_sample,
            "SUM_BY_STATE",
            [
                "What is the total paid in {a} for {b}?",
                "Total {b} payments in {a}.",
                "How much was paid in {a} for {b} payments?",
                "Sum of {b} payments made in {a}.",
            ],
        ),
        (
            "STATE", two_filter_state_sample, "NATURE", two_filter_nature_sample,
            "COUNT_BY_STATE",
            [
                "How many {b} payments were made in {a}?",
                "Count {b} payments in {a}.",
            ],
        ),
        (
            "STATE", two_filter_state_sample, "CATEGORY", two_filter_category_sample,
            "SUM_BY_STATE",
            [
                "What is the total paid in {a} for {b}?",
                "Total payments in {a} for the {b} category.",
            ],
        ),
        (
            "COMPANY", two_filter_company_sample, "NATURE", two_filter_nature_sample,
            "SUM_BY_COMPANY",
            [
                "How much did {a} pay for {b}?",
                "Total {b} payments made by {a}.",
                "Sum of {b} payments from {a}.",
            ],
        ),
        (
            "COMPANY", two_filter_company_sample, "STATE", two_filter_state_sample,
            "COUNT_BY_COMPANY",
            [
                "How many payments did {a} make in {b}?",
                "Count payments by {a} in {b}.",
            ],
        ),
    ]

    for type_a, sample_a, type_b, sample_b, intent, templates in two_filter_combos:
        for value_a in sample_a:
            for value_b in sample_b:
                template = rng.choice(templates)
                question = template.format(a=value_a, b=value_b)
                add_multi(question, intent, [(type_a, value_a), (type_b, value_b)])

    # COMPARE: two DIFFERENT values of the SAME entity type in one question,
    # e.g. "Did NY or TX receive more total payments?" Sampled at similar
    # scale to the two-filter combos above (pairs, not full cross product,
    # to keep dataset size in check).
    compare_templates = [
        "Did {a} or {b} receive more total payments?",
        "Which had more payments, {a} or {b}?",
        "Compare total paid in {a} vs {b}.",
        "Is {a} or {b} higher in total payments?",
        "Which received more overall, {a} or {b}?",
    ]
    compare_configs = [
        ("STATE", "COMPARE_BY_STATE", two_filter_state_sample, 40),
        ("COMPANY", "COMPARE_BY_COMPANY", two_filter_company_sample, 40),
        ("NATURE", "COMPARE_BY_NATURE", two_filter_nature_sample, 40),
        ("CATEGORY", "COMPARE_BY_CATEGORY", two_filter_category_sample, 40),
    ]
    for entity_type, intent, value_pool, num_pairs in compare_configs:
        pool = list(value_pool)
        if len(pool) < 2:
            continue
        for _ in range(min(num_pairs, len(pool) * (len(pool) - 1) // 2)):
            value_a, value_b = rng.sample(pool, 2)
            template = rng.choice(compare_templates)
            question = template.format(a=value_a, b=value_b)
            add_multi(question, intent, [(entity_type, value_a), (entity_type, value_b)])

    groupby_templates = [
        ("Show total payments broken down by state.", "GROUP_SUM_BY_STATE"),
        ("Which state received the most total payments?", "GROUP_SUM_BY_STATE"),
        ("Group total payments by recipient state.", "GROUP_SUM_BY_STATE"),
        ("Break down the total dollar amount by state.", "GROUP_SUM_BY_STATE"),
        ("I want to see payments summed per state.", "GROUP_SUM_BY_STATE"),
        ("Give me a state-by-state total.", "GROUP_SUM_BY_STATE"),
        ("Show total amount paid by nature of payment.", "GROUP_SUM_BY_NATURE"),
        ("Break down total payments by nature of payment.", "GROUP_SUM_BY_NATURE"),
        ("Group the total dollar amount by payment nature.", "GROUP_SUM_BY_NATURE"),
        ("How does the total spend break down by payment type?", "GROUP_SUM_BY_NATURE"),
        ("What is the average payment amount by nature of payment?", "GROUP_AVG_BY_NATURE"),
        ("Show average payment broken down by nature of payment.", "GROUP_AVG_BY_NATURE"),
        ("Break down the average payment size by payment nature.", "GROUP_AVG_BY_NATURE"),
        ("What's the mean payment per nature of payment?", "GROUP_AVG_BY_NATURE"),
        ("Show total payments broken down by product category.", "GROUP_SUM_BY_CATEGORY"),
        ("Break down payments by therapeutic category.", "GROUP_SUM_BY_CATEGORY"),
        ("Group the total dollar amount by product category.", "GROUP_SUM_BY_CATEGORY"),
        ("How does total spend break down across categories?", "GROUP_SUM_BY_CATEGORY"),
    ]
    for _ in range(120):
        q, intent = rng.choice(groupby_templates)
        add(q, intent)

    # Reserve room for the fixed SUM/COUNT/AVG_TOTAL floor added below.
    overall_floor = OVERALL_PER_INTENT * 3
    main_target = max(0, target_size - overall_floor) if target_size and target_size > 0 else 0

    rng.shuffle(examples)

    if main_target:
        if len(examples) > main_target:
            examples = examples[:main_target]
        else:
            original = list(examples)
            while len(examples) < main_target:
                examples.append(rng.choice(original))

    examples.extend(build_overall_examples(rng))
    rng.shuffle(examples)

    return examples


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    target_size = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TARGET

    print(f"Extracting entity vocabularies from {csv_path}...")
    vocab = extract_vocab(csv_path)
    for entity_type in ENTITY_TYPES:
        print(f"  {entity_type}: {len(vocab[entity_type])} distinct values")

    vocab["PROVIDER_TYPE_ALIASES"] = PROVIDER_TYPE_ALIASES

    Path(VOCAB_OUT).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vocab, VOCAB_OUT)
    print(f"Saved entity vocabularies to {VOCAB_OUT}")

    rng = random.Random(42)
    dataset = build_dataset(vocab, target_size, rng)

    Path(OUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example) + "\n")

    intents = sorted(set(ex["intent"] for ex in dataset))
    print(f"\nGenerated {len(dataset)} examples across {len(intents)} intents.")
    print(f"Saved to {OUT_JSONL}")


if __name__ == "__main__":
    main()
