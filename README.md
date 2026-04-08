# Plagarism-or-word-similarity-detector-using-lstm-
This project implements a **Siamese LSTM network** to detect textual similarity and potential plagiarism between two sentences.   Instead of analyzing sentences independently, the model learns a **shared representation** and compares them to determine semantic similarity.  ---
# siamese-lstm-plagiarism-detector  
### Written by Kaustav Banerjee

---

## 📌 Overview
This project implements a **Siamese LSTM network** to detect textual similarity and potential plagiarism between two sentences.  
Instead of analyzing sentences independently, the model learns a **shared representation** and compares them to determine semantic similarity.

---

## 📊 Dataset Description

The dataset (`train_snli.txt`) contains sentence pairs with labels indicating similarity.

### Structure:
- **source_text** → Original sentence  
- **plag_text** → Compared sentence  
- **label** → Binary output (1 = similar/plagiarized, 0 = not similar)

### Preprocessing Steps:
- Removed duplicate entries  
- Cleaned text:
  - Lowercasing  
  - Removing special characters  
  - Normalizing whitespace  

---

## 🧹 Text Processing Pipeline

### 1. Tokenization
- Simple whitespace-based tokenization  
- No advanced NLP (no stemming, no lemmatization)

### 2. Vocabulary Creation
- Built from entire dataset  
- Includes:
  - `<PAD>` → padding token  
  - `<OOV>` → out-of-vocabulary token  

### 3. Encoding
- Sentences converted to integer sequences  
- Unknown words mapped to `<OOV>`

### 4. Padding
- Fixed sequence length: **50**
- Short sequences → padded  
- Long sequences → truncated  

---

## ⚙️ Model Architecture

### Siamese Network Design:
Two identical LSTM branches process both sentences.

#code
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ==============================================================
# 1. LOAD & CLEAN DATA
# ==============================================================
file_path = "/content/sample_data/train_snli.txt"

df = pd.read_csv(
    file_path,
    sep="\t",
    header=None,
    names=["source_text", "plag_text", "label"]
)

print("Raw shape:", df.shape)

# Remove duplicates
df = df.drop_duplicates()
df = df.drop_duplicates(subset=["source_text", "plag_text"])
df = df.reset_index(drop=True)

print("After dedup shape:", df.shape)
print(df.head())

# ==============================================================
# 2. TEXT CLEANING
# ==============================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["source_text"] = df["source_text"].apply(clean_text)
df["plag_text"]   = df["plag_text"].apply(clean_text)

source_sentences = df["source_text"].tolist()
plag_sentences   = df["plag_text"].tolist()
labels           = df["label"].values

# ==============================================================
# 3. BUILD VOCABULARY (Tokenizer)
# ==============================================================
def tokenize(sentence):
    return sentence.split()

all_sentences = source_sentences + plag_sentences

word_counter = Counter()
for sent in all_sentences:
    word_counter.update(tokenize(sent))

PAD_TOKEN = "<PAD>"
OOV_TOKEN = "<OOV>"

word2idx = {PAD_TOKEN: 0, OOV_TOKEN: 1}
for word in word_counter:
    word2idx[word] = len(word2idx)

vocab_size = len(word2idx)
print("Vocabulary size:", vocab_size)

# ==============================================================
# 4. SENTENCES → SEQUENCES → PADDED TENSORS
# ==============================================================
MAX_LEN = 50  # same as reference (max_len=50)

def sentence_to_indices(sentence):
    return [word2idx.get(w, word2idx[OOV_TOKEN]) for w in tokenize(sentence)]

def pad_to_max(sequences, max_len):
    padded = []
    for seq in sequences:
        seq = torch.tensor(seq, dtype=torch.long)
        if len(seq) < max_len:
            pad = torch.zeros(max_len - len(seq), dtype=torch.long)
            seq = torch.cat([seq, pad])
        else:
            seq = seq[:max_len]
        padded.append(seq)
    return torch.stack(padded)

source_seqs = [sentence_to_indices(s) for s in source_sentences]
plag_seqs   = [sentence_to_indices(s) for s in plag_sentences]

X1 = pad_to_max(source_seqs, MAX_LEN)
X2 = pad_to_max(plag_seqs,   MAX_LEN)
y  = torch.tensor(labels, dtype=torch.float32)

print("X1 shape:", X1.shape)
print("X2 shape:", X2.shape)
print("y  shape:", y.shape)

# ==============================================================
# 5. SPLIT DATA  (mirrors: train_test_split)
# ==============================================================
indices = list(range(len(y)))

train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

X1_train, X1_test = X1[train_idx], X1[test_idx]
X2_train, X2_test = X2[train_idx], X2[test_idx]
y_train,  y_test  = y[train_idx],  y[test_idx]

print("Train size:", len(y_train))
print("Test  size:", len(y_test))

# ==============================================================
# 6. DATASET & DATALOADER
# ==============================================================
class PlagiarismDataset(Dataset):
    def __init__(self, X1, X2, y):
        self.X1 = X1
        self.X2 = X2
        self.y  = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]

train_dataset = PlagiarismDataset(X1_train, X2_train, y_train)
test_dataset  = PlagiarismDataset(X1_test,  X2_test,  y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# ==============================================================
# 7. BUILD SIAMESE LSTM MODEL
#    Architecture mirrors reference:
#    Embedding → LSTM → Subtract → Dense(64,relu) → Dropout(0.5) → Dense(1,sigmoid)
# ==============================================================
class SiameseNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, lstm_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embedding_dim, lstm_dim, batch_first=True)
        self.dense     = nn.Linear(lstm_dim, 64)
        self.dropout   = nn.Dropout(0.5)
        self.out       = nn.Linear(64, 1)

    def encode(self, x):
        embedded = self.embedding(x)                          # (B, L, emb)
        _, (h, _) = self.lstm(embedded)                       # h: (1, B, lstm_dim)
        return h.squeeze(0)                                   # (B, lstm_dim)

    def forward(self, x1, x2):
        h1 = self.encode(x1)                                  # (B, lstm_dim)
        h2 = self.encode(x2)                                  # (B, lstm_dim)
        diff = h1 - h2                                        # Subtract (like Keras Subtract)
        out  = F.relu(self.dense(diff))                       # Dense(64, relu)
        out  = self.dropout(out)                              # Dropout(0.5)
        out  = torch.sigmoid(self.out(out)).squeeze(1)        # Dense(1, sigmoid)
        return out                                            # (B,)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = SiameseNetwork(vocab_size, embedding_dim=128, lstm_dim=64).to(device)
print(model)

# Count parameters (mirrors model.summary())
total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params:,}")

# ==============================================================
# 8. LOSS & OPTIMIZER
# ==============================================================
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==============================================================
# 9. TRAIN MODEL  (with early stopping like EarlyStopping(patience=2))
# ==============================================================
EPOCHS   = 10
PATIENCE = 2

history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

best_val_loss    = float("inf")
patience_counter = 0
best_weights     = None

# Use 20% of train set as validation (mirrors validation_split=0.2)
val_split  = int(0.8 * len(train_dataset))
train_sub  = torch.utils.data.Subset(train_dataset, range(val_split))
val_sub    = torch.utils.data.Subset(train_dataset, range(val_split, len(train_dataset)))

train_sub_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
val_loader       = DataLoader(val_sub,   batch_size=32, shuffle=False)

for epoch in range(EPOCHS):
    # ---- Train ----
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x1, x2, yb in train_sub_loader:
        x1, x2, yb = x1.to(device), x2.to(device), yb.to(device)

        optimizer.zero_grad()
        preds = model(x1, x2)
        loss  = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(yb)
        correct    += ((preds > 0.5).float() == yb).sum().item()
        total      += len(yb)

    train_loss = total_loss / total
    train_acc  = correct / total

    # ---- Validate ----
    model.eval()
    val_loss_total, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for x1, x2, yb in val_loader:
            x1, x2, yb = x1.to(device), x2.to(device), yb.to(device)
            preds = model(x1, x2)
            loss  = criterion(preds, yb)
            val_loss_total += loss.item() * len(yb)
            val_correct    += ((preds > 0.5).float() == yb).sum().item()
            val_total      += len(yb)

    val_loss = val_loss_total / val_total
    val_acc  = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} — "
          f"accuracy: {train_acc:.4f} — loss: {train_loss:.4f} — "
          f"val_accuracy: {val_acc:.4f} — val_loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# Restore best weights
if best_weights:
    model.load_state_dict(best_weights)
    print("Best weights restored.")

# ==============================================================
# 10. EVALUATION
# ==============================================================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for x1, x2, yb in test_loader:
        x1, x2 = x1.to(device), x2.to(device)
        preds = model(x1, x2).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
pred_label = (all_preds > 0.5).astype(int)

print("Accuracy:", accuracy_score(all_labels, pred_label))
print(classification_report(all_labels, pred_label))

# ==============================================================
# 11. PLOT TRAINING CURVES
# ==============================================================
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["val_acc"],   label="val_acc")
plt.title("Accuracy over epochs")
plt.legend()
plt.show()

plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"],   label="val_loss")
plt.title("Loss over epochs")
plt.legend()
plt.show()

# ==============================================================
# 12. SAMPLE PREDICTION ON TEST SET
# ==============================================================
model.eval()
with torch.no_grad():
    src = X1_test[0:1].to(device)
    plg = X2_test[0:1].to(device)
    score = model(src, plg).item()
    print("Pred:", score)

# ==============================================================
# 13. PREDICT ON USER INPUT  (mirrors predict_pair())
# ==============================================================
def predict_pair(source_text, plag_text, word2idx, max_len, model, device):
    def to_tensor(text):
        indices = [word2idx.get(w, word2idx[OOV_TOKEN]) for w in tokenize(clean_text(text))]
        seq = torch.tensor(indices, dtype=torch.long)
        if len(seq) < max_len:
            pad = torch.zeros(max_len - len(seq), dtype=torch.long)
            seq = torch.cat([seq, pad])
        else:
            seq = seq[:max_len]
        return seq.unsqueeze(0).to(device)   # (1, max_len)

    model.eval()
    with torch.no_grad():
        src_tensor = to_tensor(source_text)
        plg_tensor = to_tensor(plag_text)
        pred = model(src_tensor, plg_tensor).item()

    print("Prediction score:", pred)
    if pred > 0.5:
        print("✅ Likely PLAGIARIZED")
    else:
        print("❌ Likely NOT plagiarized")
    return pred
