import csv
import random
import sys

TARGET_SIZE = 1000
BUCKET_CAP = 3
MAX_POOL_SIZE = 20000

STRATIFY_FIELDS = [
    "Recipient_State",
    "Covered_Recipient_Type",
    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
    "Nature_of_Payment_or_Transfer_of_Value",
]

PROGRESS_EVERY = 500_000


def bucket_key(row):
    return "|".join(row.get(field) or "UNKNOWN" for field in STRATIFY_FIELDS)


def main():
    if len(sys.argv) < 2:
        print("Usage: python train/sample_data.py <input-csv> [output-csv]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "train/stratified_sample.csv"

    print(f"Streaming {in_path} and stratifying data...")

    rng = random.Random(42)

    total_rows_scanned = 0
    valid_row_count = 0
    buckets = {}
    candidates = []
    fieldnames = None

    with open(in_path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            total_rows_scanned += 1
            if total_rows_scanned % PROGRESS_EVERY == 0:
                print(f"  scanned {total_rows_scanned:,} rows...")

            key = bucket_key(row)
            current_count = buckets.get(key, 0)

            if current_count < BUCKET_CAP:
                buckets[key] = current_count + 1
                valid_row_count += 1

                if len(candidates) < MAX_POOL_SIZE:
                    candidates.append(row)
                else:
                    r = rng.randrange(valid_row_count)
                    if r < MAX_POOL_SIZE:
                        candidates[r] = row

    print(f"Scan complete. Scanned {total_rows_scanned:,} rows.")
    print(f"Collected valid candidates across {len(buckets):,} composite buckets.")

    if not candidates:
        print("No valid rows found.")
        return

    rng.shuffle(candidates)
    final_set = candidates[:TARGET_SIZE]

    final_states = set()
    final_companies = set()
    final_natures = set()
    final_types = set()

    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for row in final_set:
            writer.writerow(row)

            final_states.add(row.get("Recipient_State"))
            final_companies.add(
                row.get("Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name")
            )
            final_natures.add(row.get("Nature_of_Payment_or_Transfer_of_Value"))
            final_types.add(row.get("Covered_Recipient_Type"))

    print("\n=== Stratified Sample Generated ===")
    print(f"Output File        : {out_path}")
    print(f"Final Row Count    : {len(final_set)}")
    print(f"Distinct States    : {len(final_states)}")
    print(f"Distinct Companies : {len(final_companies)}")
    print(f"Distinct Categories: {len(final_natures)}")
    print(f"Distinct Types     : {len(final_types)}")
    print("===================================\n")


if __name__ == "__main__":
    main()
