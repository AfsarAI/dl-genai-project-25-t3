"""
scripts/predict_transformer.py

Usage:
python scripts/predict_transformer.py --test_csv data/raw/test.csv --model_dir models/roberta-run --out submission.csv --model_name roberta-base
"""

import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import os

LABEL_COLS = ["anger","fear","joy","sadness","surprise"]

class TextDatasetPred(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {k:v.squeeze(0) for k,v in encoding.items()}

class TransformerForMultiLabelInfer(torch.nn.Module):
    def __init__(self, model_name, n_labels, dropout=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        hidden = self.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden, n_labels)
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(2)
        summed = (last_hidden * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        pooled = summed / counts
        x = self.dropout(pooled)
        logits = self.classifier(x)
        return logits

def predict_proba(model, loader, device):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(logits.cpu().numpy())
    return np.vstack(all_logits)

def find_best_thresholds(val_probs, val_labels):
    # find a single best global threshold by scanning 0.2..0.8 (or do per-label)
    best_t = 0.5; best_f1 = -1
    for t in np.linspace(0.2, 0.8, 31):
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(val_labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_t = t
    return best_t, best_f1

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # load model
    model = TransformerForMultiLabelInfer(args.model_name, n_labels=len(LABEL_COLS), dropout=args.dropout)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "best_model.pt"), map_location=device))
    model.to(device)

    # if validation provided, use it for threshold tuning
    val_probs = None; val_labels = None
    if args.val_csv:
        val_df = pd.read_csv(args.val_csv)
        val_texts = val_df['text'].fillna("").astype(str).values
        val_labels = val_df[LABEL_COLS].values
        val_ds = TextDatasetPred(val_texts, tokenizer, max_len=args.max_len)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        val_logits = predict_proba(model, val_loader, device)
        val_probs = 1/(1+np.exp(-val_logits))

    # test
    test_df = pd.read_csv(args.test_csv)
    test_texts = test_df['text'].fillna("").astype(str).values
    test_ds = TextDatasetPred(test_texts, tokenizer, max_len=args.max_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_logits = predict_proba(model, test_loader, device)
    test_probs = 1/(1+np.exp(-test_logits))

    # threshold
    threshold = args.threshold
    if args.val_csv and args.tune_threshold:
        best_t, best_f1 = find_best_thresholds(val_probs, val_labels)
        print("Tuned threshold:", best_t, "f1:", best_f1)
        threshold = best_t

    preds = (test_probs >= threshold).astype(int)
    submission = pd.DataFrame({"id": test_df["id"]})
    for i, col in enumerate(LABEL_COLS):
        submission[col] = preds[:, i]
    submission.to_csv(args.out, index=False)
    print("Saved:", args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="data/raw/test.csv")
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default="models/roberta-run")
    parser.add_argument("--out", type=str, default="submission.csv")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune_threshold", action="store_true")
    args = parser.parse_args()
    main(args)
