import os
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score # Added accuracy
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn

try:
    import wandb
    from kaggle_secrets import UserSecretsClient
    WANDB_OK = True
except:
    WANDB_OK = False

LABEL_COLS = ["anger","fear","joy","sadness","surprise"]

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {k:v.squeeze(0) for k,v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

class GenericTransformer(nn.Module):
    def __init__(self, model_path, num_labels, dropout=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, config=self.config, local_files_only=True)
        h = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h, num_labels)

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask)
        last = out.last_hidden_state     # universal
        mask_exp = mask.unsqueeze(-1)
        pooled = (last * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
        return self.classifier(self.dropout(pooled))

def kaggle_wandb_init(project, run_name, args):
    if not WANDB_OK:
        return
    try:
        key = UserSecretsClient().get_secret("WANDB_API_KEY")
        wandb.login(key=key)
        wandb.init(project=project, name=run_name, config=vars(args), reinit=True)
    except:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project=project, name=run_name, config=vars(args), reinit=True)

def train_epoch(model, loader, opt, sch, device):
    model.train()
    losses = []
    for batch in tqdm(loader, leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        opt.zero_grad()
        logits = model(ids, mask)
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()
        opt.step()
        sch.step()
        losses.append(loss.item())
    return np.mean(losses)

def val_epoch(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    val_losses = []
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ids, mask)
            loss = criterion(logits, labels)
            val_losses.append(loss.item())

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)

    probs = 1/(1+np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels.flatten(), preds.flatten())
    avg_loss = np.mean(val_losses)

    return avg_loss, f1, acc

def main(args):
    df = pd.read_csv(args.train_csv)
    train_df, val_df = train_test_split(df, test_size=args.val_frac, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.local_model_path, local_files_only=True)

    train_ds = TextDataset(train_df["text"].values, train_df[LABEL_COLS].values, tokenizer, args.max_len)
    val_ds   = TextDataset(val_df["text"].values,   val_df[LABEL_COLS].values, tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GenericTransformer(args.local_model_path, len(LABEL_COLS), args.dropout).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06*steps), steps)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb_project:
        kaggle_wandb_init(args.wandb_project, args.run_name, args)

    best_f1 = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_f1, val_acc = val_epoch(model, val_loader, device)

        print(f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val F1={val_f1:.4f} | Val Acc={val_acc:.4f}")

        if args.wandb_project and WANDB_OK:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1": val_f1
            })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pt")
            print("Saved BEST!")

    if args.wandb_project and WANDB_OK:
        wandb.finish()

    print("\nBest F1:", best_f1)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--local_model_path", type=str)
    p.add_argument("--output_dir", type=str)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--run_name", type=str, default="transformer-run")
    args = p.parse_args()
    main(args)
