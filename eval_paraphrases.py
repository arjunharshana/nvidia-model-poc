"""
python eval_paraphrases.py [path-to-jsonl]
Evaluates the trained model against a hand-written paraphrase set, separate
from the templated train/val split reported by train_cms_qa.py.
"""

import json
import sys

import joblib
import torch

from train_cms_qa import IntentEntityModel, encode_question, normalize_example

MODEL_PATH = "cms_qa_model.pt"
VOCAB_PATH = "cms_qa_word_vocab.joblib"
INTENT_ENCODER_PATH = "cms_qa_intent_encoder.joblib"
ENTITY_TYPE_ENCODER_PATH = "cms_qa_entity_type_encoder.joblib"
DEFAULT_EVAL_PATH = "train/paraphrase_eval.jsonl"


def load_artifacts():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    word2idx = joblib.load(VOCAB_PATH)
    intent_encoder = joblib.load(INTENT_ENCODER_PATH)
    entity_types = joblib.load(ENTITY_TYPE_ENCODER_PATH)
    threshold = checkpoint.get("entity_type_threshold", 0.5)

    model = IntentEntityModel(
        vocab_size=checkpoint["vocab_size"],
        num_intents=checkpoint["num_intents"],
        num_entity_types=checkpoint["num_entity_types"],
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, word2idx, intent_encoder, entity_types, threshold


def predict(question, model, word2idx, intent_encoder, entity_types, threshold):
    ids = encode_question(question, word2idx)
    tokens = torch.tensor(ids, dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        intent_logits, entity_type_logits = model(tokens, offsets)

    intent = intent_encoder.inverse_transform([intent_logits.argmax(dim=1).item()])[0]
    probs = torch.sigmoid(entity_type_logits).squeeze(0)
    predicted_types = {entity_types[i] for i, p in enumerate(probs) if p.item() >= threshold}
    return intent, predicted_types


def main():
    eval_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EVAL_PATH

    model, word2idx, intent_encoder, entity_types, threshold = load_artifacts()

    examples = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(normalize_example(json.loads(line)))

    intent_correct = 0
    entity_type_correct = 0
    both_correct = 0
    mistakes = []

    for ex in examples:
        gold_types = {t for t, v in ex["entities"]}
        pred_intent, pred_types = predict(
            ex["question"], model, word2idx, intent_encoder, entity_types, threshold
        )

        intent_ok = pred_intent == ex["intent"]
        entity_ok = pred_types == gold_types

        intent_correct += intent_ok
        entity_type_correct += entity_ok
        both_correct += intent_ok and entity_ok

        if not (intent_ok and entity_ok):
            mistakes.append((ex["question"], ex["intent"], pred_intent, gold_types, pred_types))

    total = len(examples)
    print(f"Paraphrase set: {eval_path} ({total} examples)")
    print(f"Intent accuracy:            {100*intent_correct/total:.2f}%")
    print(f"Entity-type set accuracy:   {100*entity_type_correct/total:.2f}%")
    print(f"Joint accuracy:             {100*both_correct/total:.2f}%")

    if mistakes:
        print(f"\nMistakes ({len(mistakes)}):")
        for question, gold_intent, pred_intent, gold_types, pred_types in mistakes:
            print(f"  Q: {question}")
            print(f"     intent: gold={gold_intent!r} pred={pred_intent!r}")
            print(f"     entity_types: gold={gold_types} pred={pred_types}")


if __name__ == "__main__":
    main()
