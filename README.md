# CMS Open Payments Q&A

A narrow, from-scratch PyTorch model that answers structured questions about the CMS Open Payments dataset (e.g. *"total paid in CA"*, *"how many payments for Oncology"*) via question → intent + entity classification → deterministic CSV aggregation, instead of LLM-generated SQL.

## Architecture

This pipeline trains a small classifier and pairs it with DuckDB for the actual data work:

1. **Stratified sampling** (`train/sample_data.py`): streams the full CMS CSV without loading it into memory, sampling 1000 rows stratified by state, provider type, company, and nature of payment.
2. **Synthetic dataset generation** (`train/generate_qa_dataset.py`): pulls the real categorical vocabulary out of that sample and synthesizes labeled `(question, intent, entity_type, entity_value)` examples across 23 intents — aggregates, filtered aggregates, group-by breakdowns, and similarity lookups.
3. **Training** (`train_cms_qa.py`): trains an `EmbeddingBag` (bag-of-words) → MLP trunk → two classification heads (intent, entity-type) model from scratch on the synthetic questions. 
4. **Inference** (`predict_cms_qa.py`): classifies the question, then either runs a DuckDB `SUM`/`COUNT`/`AVG`/`GROUP BY` query directly over the CSV, or, for `SIMILAR_COMPANIES_TO` / `SIMILAR_STATES_TO`, hands off to `similarity.py`, which builds a payment-nature profile per entity and ranks matches by cosine similarity.

## Setup

Check your NVIDIA driver:
```bash
nvidia-smi
```

Create a virtual environment and install dependencies:
```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Verify CUDA and Tensor Cores are working:
```bash
python check_gpu.py
```

Place a CMS Open Payments CSV in `data/`.

## Run

Generate the stratified sample from your full CSV:
```bash
python train/sample_data.py <path-to-cms-csv> train/stratified_sample.csv
```
Writes `train/stratified_sample.csv` (1000 rows).

Build the synthetic training dataset from that sample:
```bash
python train/generate_qa_dataset.py
```
Writes `train/qa_dataset.jsonl` (~6000 labeled question examples) and `data/entity_vocab.joblib` (known categorical values per entity type).

Train the model from scratch:
```bash
python train_cms_qa.py
```
Writes `cms_qa_model.pt` (model weights), `cms_qa_word_vocab.joblib` (question tokenizer vocab), and `cms_qa_intent_encoder.joblib` / `cms_qa_entity_type_encoder.joblib` (label encoders).

Ask a question:
```bash
python predict_cms_qa.py "What is the total paid in CA?"
```
Loads the artifacts above and prints an answer — nothing new is written to disk.

Other usage:
```bash
python predict_cms_qa.py                                                    # interactive
python predict_cms_qa.py "..." --csv data/OP_DTL_GNRL_PGYR2025_...csv       # specific CSV
python predict_cms_qa.py "Which company is similar to Vitalant?"            
