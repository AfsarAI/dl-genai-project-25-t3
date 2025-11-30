import os
import argparse
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

# Kaggle W&B safe mode
try:
    import wandb
    from kaggle_secrets import UserSecretsClient
    WANDB_OK = True
except:
    WANDB_OK = False

LABEL_COLS = ["anger","fear","joy","sadness","surprise"]

LOCAL_MODEL = "/kaggle/working/roberta-base-local"   # <---- FIX

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

class TransformerForMultiLabel(nn.Module):
    def __init__(self, model_dir, n_labels, dropout=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.backbone = AutoModel.from_pretrained(model_dir, config=self.config, local_files_only=True)

        h = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h, n_labels)

    def forward(self, ids, mask):
        out = self.backbone(input_ids=ids, attention_mask=mask)
        last = out.last_hidden_state
        m = mask.unsqueeze(2)
        pooled = (last * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.classifier(self.dropout(pooled))

def train_epoch(model, loader, optimizer, scheduler, device, scaler=None):
    model.train()
    losses = []
    for batch in tqdm(loader, leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(ids, mask)
                loss = nn.BCEWithLogitsLoss()(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(ids, mask)
            loss = nn.BCEWithLogitsLoss()(logits, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        losses.append(loss.item())

    return np.mean(losses)

def valid_epoch(model, loader, device, threshold=0.5):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            labels = batch["labels"].cpu().numpy()
            logits = model(ids, mask).cpu().numpy()

            all_logits.append(logits)
            all_labels.append(labels)

    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return f1

def kaggle_wandb_init(project, run_name, args):
    if not WANDB_OK:
        return None

    try:
        key = UserSecretsClient().get_secret("WANDB_API_KEY")
        wandb.login(key=key)
        return wandb.init(project=project, name=run_name, config=vars(args), reinit=True)
    except:
        os.environ["WANDB_MODE"] = "offline"
        return wandb.init(project=project, name=run_name, config=vars(args))

def main(args):
    df = pd.read_csv(args.train_csv)
    train_df, val_df = train_test_split(df, test_size=args.val_frac, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL, local_files_only=True)

    train_texts = train_df["text"].astype(str).values
    val_texts = val_df["text"].astype(str).values

    train_labels = train_df[LABEL_COLS].values
    val_labels = val_df[LABEL_COLS].values

    train_ds = TextDataset(train_texts, train_labels, tokenizer, args.max_len)
    val_ds = TextDataset(val_texts, val_labels, tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerForMultiLabel(LOCAL_MODEL, len(LABEL_COLS), args.dropout).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.06), total_steps)

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb_project:
        kaggle_wandb_init(args.wandb_project, args.run_name, args)

    best_f1 = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        val_f1 = valid_epoch(model, val_loader, device, args.threshold)

        print(f"Loss={train_loss:.4f}  |  Val F1={val_f1:.4f}")

        if args.wandb_project:
            wandb.log({"train_loss": train_loss, "val_f1": val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print("Saved BEST!")

    print("\nBest Val F1:", best_f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/roberta-run")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="run-roberta")
    args = parser.parse_args()
    main(args)
