# 🧠 Multi-Label Emotion Classification: Deep Learning & GenAI Project
### Course: IITM BS Degree - Deep Learning Practice (23f2002023)

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![WandB](https://img.shields.io/badge/WandB-Tracking-orange)

## 📖 Project Overview
This project focuses on **Multi-Label Text Classification** to detect emotions in text. Unlike standard classification, a single text can have multiple emotions simultaneously (e.g., both 'Sadness' and 'Fear').

The goal is to build a robust pipeline comparing a **Baseline Model (Scratch)** against **State-of-the-Art Transformer Models**, and finally boosting performance using **Ensemble Learning**.

**Emotions Tracked:** `anger`, `fear`, `joy`, `sadness`, `surprise`

---

## 📂 Project Structure & File Description

The codebase is modular, separating preprocessing, training, and inference logic for clarity.

    .
    ├── scripts/                                 # All training / inference code
    │   │
    │   ├── scratch/                              # Baseline Approach (TF-IDF + MLP)
    │   │     ├── train.py                        # Train the MLP model
    │   │     └── predict.py                      # Predict using saved MLP
    │   │
    │   ├── pretrained/                           # Transformer Fine-tuning
    │   │     ├── train_transformer.py            # Fine-tune BERT / RoBERTa
    │   │     └── predict_transformer.py          # Transformer inference logic
    │   │
    │   └── ensemble/                             # Model Averaging (Ensemble)
    │         └── train_predict_all.py            # Train multiple models & combine
    │
    ├── models/                                   # Saved checkpoints (.pth / .pt)
    │
    ├── data/                                     # Raw train/test CSV datasets
    │
    ├── requirements.txt                          # Python dependencies
    │
    └── README.md                                 # Project documentation

---

## 🚀 Methodology & Approaches

### Approach 1: The Baseline (From Scratch) 🏗️
* **Technique:** TF-IDF (Term Frequency-Inverse Document Frequency) + MLP (Multi-Layer Perceptron)
* **Architecture:**
    * N-gram vectors (1-gram & 2-gram)
    * 3 Linear layers + ReLU
    * Dropout (0.3)
* **Purpose:** Create a basic benchmark for comparison.

---

### Approach 2: Transfer Learning (Pretrained Transformers) 🤖
**Models Used:**
- BERT (`bert-base-uncased`)
- RoBERTa (`roberta-base`)
- DistilRoBERTa (`distilroberta-base`)

**Fine-Tuning:**
- Custom classification head
- Loss: `BCEWithLogitsLoss`
- Optimizer: `AdamW`

Transforms understand deeper context and semantics compared to TF-IDF.

---

### Approach 3: Ensemble Learning (The Booster) 🤝
**Soft Voting (Probability Averaging)**

`P_final = (P_bert + P_roberta + P_distil) / 3`

Provides consistently higher F1 scores by reducing variance.

---

## 📊 Experiment Tracking (Weights & Biases)
Tracked:
- Train/Val Loss
- Macro F1 Score
- Accuracy

WandB helps visualize how transformers outperform baseline and how ensemble performs best.

---

## 💻 How to Run

### 1. Setup Environment
```bash
pip install -r requirements.txt
```
    
### 2. Run Baseline (Scratch)
```bash
python scripts/scratch/train.py --epochs 10 --use_wandb
python scripts/scratch/predict.py
```
    
### 3. Run Pretrained Transformers
```bash
python scripts/pretrained/train_transformer.py \
    --local_model_path /path/to/roberta \
    --run_name roberta-experiment
```

### 4. Run Ensemble (Full Pipeline)
```bash
python scripts/ensemble/train_predict_all.py \
    --train_csv data/train.csv \
    --test_csv data/test.csv \
    --wandb_project DL-GenAI-Ensemble
```

## 📈 Results & Observations

| Model    | Technique    | Performance          |
| -------- | ------------ | -------------------- |
| Scratch  | TF-IDF + MLP | Baseline             |
| BERT     | Transformer  | High Accuracy        |
| RoBERTa  | Transformer  | Higher Accuracy      |
| Ensemble | Voting       | **Best F1 Score 🏆** |

---
    
##### Created by **Mohd Afsar** for **DL-GenAI Course** (23f2002023)
