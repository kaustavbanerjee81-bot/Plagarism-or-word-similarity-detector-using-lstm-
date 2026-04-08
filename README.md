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
