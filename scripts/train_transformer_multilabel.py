"""
scripts/train_transformer_multilabel.py

Usage example:
python scripts/train_transformer_multilabel.py \
  --train_csv data/raw/train.csv \
  --val_frac 0.1 \
  --model_name roberta-base \
  --output_dir models/roberta-base-run1 \
  --epochs 3 \
  --batch_size 8 \
  --lr 2e-5 \
  --wandb_project 23f2002023-t32025 \
  --run_name roberta-baseline
"""

import os
import argparse
from datetime import datetime
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn

import wandb

LABEL_COLS = ["anger","fear","joy","sadness","surprise"]

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {k:v.squeeze(0) for k,v in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

class TransformerForMultiLabel(nn.Module):
    def __init__(self, model_name, n_labels, dropout=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, n_labels)
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # use pooled output: for RoBERTa/BERT it's last_hidden_state[:,0,:] or `pooler_output` for BERT
        # safer: mean pooling
        last_hidden = out.last_hidden_state  # (B, L, H)
        mask = attention_mask.unsqueeze(2)  # (B, L, 1)
        summed = (last_hidden * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        pooled = summed / counts
        x = self.dropout(pooled)
        logits = self.classifier(x)
        return logits

def compute_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def train_epoch(model, loader, optimizer, scheduler, device, scaler=None):
    model.train()
    losses = []
    for batch in tqdm(loader, leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = nn.BCEWithLogitsLoss()(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.BCEWithLogitsLoss()(logits, labels)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
    return np.mean(losses)

def valid_epoch(model, loader, device, threshold=0.5):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = logits.cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels)
    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)
    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= threshold).astype(int)
    macro_f1 = compute_macro_f1(all_labels, preds)
    return macro_f1, all_labels, probs

def main(args):
    seed_everything(args.seed)
    df = pd.read_csv(args.train_csv)
    # if val_frac > 0 use split
    if args.val_csv is None:
        train_df, val_df = train_test_split(df, test_size=args.val_frac, random_state=args.seed, shuffle=True)
    else:
        train_df = df
        val_df = pd.read_csv(args.val_csv)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_texts = train_df['text'].fillna("").astype(str).values
    val_texts = val_df['text'].fillna("").astype(str).values
    train_labels = train_df[LABEL_COLS].values
    val_labels = val_df[LABEL_COLS].values

    train_ds = TextDataset(train_texts, train_labels, tokenizer, max_len=args.max_len)
    val_ds = TextDataset(val_texts, val_labels, tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerForMultiLabel(args.model_name, n_labels=len(LABEL_COLS), dropout=args.dropout)
    model = model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    t_total = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*t_total), num_training_steps=t_total)

    scaler = torch.cuda.amp.GradScaler() if args.use_amp and torch.cuda.is_available() else None

    # W&B init
    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    best_f1 = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler=scaler)
        val_f1, val_labels, val_probs = valid_epoch(model, val_loader, device, threshold=args.threshold)
        print(f"Train loss {train_loss:.4f} | Val Macro F1 {val_f1:.4f}")
        if args.wandb_project is not None:
            wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_macro_f1": val_f1})
        # save best
        ckpt_path = os.path.join(args.output_dir, f"model_epoch{epoch+1}_f1{val_f1:.4f}.pt")
        torch.save(model.state_dict(), ckpt_path)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print("Saved best model:", best_path)
    print("Best val macro F1:", best_f1)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="data/raw/train.csv")
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, default="models/roberta-run")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="run1")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
