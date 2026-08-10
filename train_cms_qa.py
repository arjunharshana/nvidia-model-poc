import json
import re
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

DATASET_PATH = "train/qa_dataset.jsonl"
MODEL_OUT = "cms_qa_model.pt"
VOCAB_OUT = "cms_qa_word_vocab.joblib"
INTENT_ENCODER_OUT = "cms_qa_intent_encoder.joblib"
ENTITY_TYPE_ENCODER_OUT = "cms_qa_entity_type_encoder.joblib"

EMBED_DIM = 64
TRUNK_HIDDEN = 128
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    # Bigrams give mean-pooled EmbeddingBag a signal for phrases like "how many".
    words = TOKEN_RE.findall(text.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def load_examples(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def build_word_vocab(questions, min_freq=1):
    freq = {}
    for q in questions:
        for tok in tokenize(q):
            freq[tok] = freq.get(tok, 0) + 1

    word2idx = {"<pad>": 0, "<unk>": 1}
    for word, count in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
        if count >= min_freq:
            word2idx[word] = len(word2idx)
    return word2idx


def encode_question(question, word2idx):
    ids = [word2idx.get(tok, word2idx["<unk>"]) for tok in tokenize(question)]
    if not ids:
        ids = [word2idx["<unk>"]]
    return ids


class QADataset(Dataset):
    def __init__(self, questions, intent_labels, entity_type_labels, word2idx):
        self.encoded = [encode_question(q, word2idx) for q in questions]
        self.intent_labels = intent_labels
        self.entity_type_labels = entity_type_labels

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx], self.intent_labels[idx], self.entity_type_labels[idx]


def collate_batch(batch):
    token_lists, intent_labels, entity_type_labels = zip(*batch)

    offsets = [0]
    flat_tokens = []
    for tokens in token_lists:
        flat_tokens.extend(tokens)
        offsets.append(offsets[-1] + len(tokens))
    offsets = offsets[:-1]

    return (
        torch.tensor(flat_tokens, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(intent_labels, dtype=torch.long),
        torch.tensor(entity_type_labels, dtype=torch.long),
    )


class IntentEntityModel(nn.Module):
    def __init__(self, vocab_size, num_intents, num_entity_types,
                 embed_dim=EMBED_DIM, hidden_dim=TRUNK_HIDDEN):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean", padding_idx=0)
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.intent_head = nn.Linear(hidden_dim, num_intents)
        self.entity_type_head = nn.Linear(hidden_dim, num_entity_types)

    def forward(self, tokens, offsets):
        embedded = self.embedding(tokens, offsets)
        features = self.trunk(embedded)
        return self.intent_head(features), self.entity_type_head(features)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    examples = load_examples(DATASET_PATH)
    print(f"Loaded {len(examples)} examples from {DATASET_PATH}")

    questions = [ex["question"] for ex in examples]
    intents = [ex["intent"] for ex in examples]
    entity_types = [ex["entity_type"] for ex in examples]

    word2idx = build_word_vocab(questions)
    print(f"Word vocab size: {len(word2idx)}")

    intent_encoder = LabelEncoder().fit(intents)
    entity_type_encoder = LabelEncoder().fit(entity_types)
    print(f"Intents ({len(intent_encoder.classes_)}): {list(intent_encoder.classes_)}")
    print(f"Entity types ({len(entity_type_encoder.classes_)}): {list(entity_type_encoder.classes_)}")

    intent_labels = intent_encoder.transform(intents)
    entity_type_labels = entity_type_encoder.transform(entity_types)

    (q_train, q_test,
     yi_train, yi_test,
     ye_train, ye_test) = train_test_split(
        questions, intent_labels, entity_type_labels,
        test_size=0.15, random_state=42, stratify=intent_labels,
    )

    train_ds = QADataset(q_train, yi_train, ye_train, word2idx)
    test_ds = QADataset(q_test, yi_test, ye_test, word2idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    model = IntentEntityModel(
        vocab_size=len(word2idx),
        num_intents=len(intent_encoder.classes_),
        num_entity_types=len(entity_type_encoder.classes_),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler_amp = GradScaler(enabled=use_amp)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for tokens, offsets, yi, ye in train_loader:
            tokens, offsets = tokens.to(device), offsets.to(device)
            yi, ye = yi.to(device), ye.to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                intent_logits, entity_type_logits = model(tokens, offsets)
                loss = criterion(intent_logits, yi) + criterion(entity_type_logits, ye)

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

            running_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}  loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    intent_correct = 0
    entity_type_correct = 0
    both_correct = 0
    total = 0
    with torch.no_grad():
        for tokens, offsets, yi, ye in test_loader:
            tokens, offsets = tokens.to(device), offsets.to(device)
            yi, ye = yi.to(device), ye.to(device)

            intent_logits, entity_type_logits = model(tokens, offsets)
            intent_preds = intent_logits.argmax(dim=1)
            entity_type_preds = entity_type_logits.argmax(dim=1)

            intent_correct += (intent_preds == yi).sum().item()
            entity_type_correct += (entity_type_preds == ye).sum().item()
            both_correct += ((intent_preds == yi) & (entity_type_preds == ye)).sum().item()
            total += yi.size(0)

    print(f"\nTest intent accuracy:      {100*intent_correct/total:.2f}%")
    print(f"Test entity-type accuracy: {100*entity_type_correct/total:.2f}%")
    print(f"Test joint accuracy:       {100*both_correct/total:.2f}%")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": len(word2idx),
            "num_intents": len(intent_encoder.classes_),
            "num_entity_types": len(entity_type_encoder.classes_),
            "embed_dim": EMBED_DIM,
            "hidden_dim": TRUNK_HIDDEN,
        },
        MODEL_OUT,
    )
    joblib.dump(word2idx, VOCAB_OUT)
    joblib.dump(intent_encoder, INTENT_ENCODER_OUT)
    joblib.dump(entity_type_encoder, ENTITY_TYPE_ENCODER_OUT)

    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved word vocab to {VOCAB_OUT}")
    print(f"Saved intent encoder to {INTENT_ENCODER_OUT}")
    print(f"Saved entity-type encoder to {ENTITY_TYPE_ENCODER_OUT}")


if __name__ == "__main__":
    main()
