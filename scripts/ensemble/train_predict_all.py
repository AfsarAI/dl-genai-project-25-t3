import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

# --- WANDB SETUP ---
try:
    import wandb
    from kaggle_secrets import UserSecretsClient
    WANDB_OK = True
except:
    WANDB_OK = False

LABEL_COLS = ["anger","fear","joy","sadness","surprise"]
os.environ["TRANSFORMERS_NO_ADDITIONAL_TEMPLATES"] = "1"

# -------------------------- DATASET --------------------------
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
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# -------------------------- MODEL --------------------------
class TransformerForMultiLabel(torch.nn.Module):
    def __init__(self, model_path, n_labels, dropout=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        self.backbone = AutoModel.from_pretrained(model_path, config=self.config, local_files_only=True)
        h = self.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(h, n_labels)

    def forward(self, ids, mask):
        out = self.backbone(input_ids=ids, attention_mask=mask)
        last = out.last_hidden_state
        mask = mask.unsqueeze(2)
        pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(self.dropout(pooled))

# -------------------------- TRAINING FUNCTIONS --------------------------
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    losses = []
    for batch in loader:
        optimizer.zero_grad()
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(ids, mask)
        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return np.mean(losses)

def valid_epoch(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    val_losses = []
    criterion = torch.nn.BCEWithLogitsLoss()

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

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    
    avg_loss = np.mean(val_losses)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels.flatten(), preds.flatten())

    return avg_loss, f1, acc

# -------------------------- TRAIN ONE MODEL --------------------------
def train_model(model_name, train_csv, out_dir, epochs=2):
    print(f"\n🔥 TRAINING: {model_name}")

    df = pd.read_csv(train_csv)
    tr, val = train_test_split(df, test_size=0.1, random_state=42)

    local_path = f"/kaggle/working/{model_name}-local"
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)

    tr_ds = TextDataset(tr.text.values, tr[LABEL_COLS].values, tokenizer, 128)
    val_ds = TextDataset(val.text.values, val[LABEL_COLS].values, tokenizer, 128)

    tr_loader = DataLoader(tr_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerForMultiLabel(local_path, len(LABEL_COLS)).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(tr_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * total_steps, total_steps)

    os.makedirs(out_dir, exist_ok=True)
    best_f1 = 0

    for ep in range(epochs):
        train_loss = train_epoch(model, tr_loader, optimizer, scheduler, device)
        val_loss, val_f1, val_acc = valid_epoch(model, val_loader, device)
        
        print(f"Epoch {ep+1} | Loss={train_loss:.4f} | Val F1={val_f1:.4f} | Val Acc={val_acc:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"{out_dir}/best_model.pt")

        # --- WandB Logging for EACH model in the ensemble ---
        if WANDB_OK and wandb.run is not None:
             # Use generic keys so they overlap in one chart, OR prefix if you want separate lines
             # For Comparison, using prefix is safer in a single run
            wandb.log({
                f"{model_name}_epoch": ep+1,
                f"{model_name}_train_loss": train_loss,
                f"{model_name}_val_loss": val_loss,
                f"{model_name}_val_accuracy": val_acc,
                f"{model_name}_val_f1": val_f1
            })

    print("BEST F1:", best_f1)

# -------------------------- PREDICT --------------------------
def predict_model(model_name, model_dir, test_csv, out_csv):
    print(f"\n🟩 Predicting with {model_name}")

    local_path = f"/kaggle/working/{model_name}-local"
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)

    model = TransformerForMultiLabel(local_path, len(LABEL_COLS))
    model.load_state_dict(torch.load(f"{model_dir}/best_model.pt", map_location="cpu"))
    model.eval()

    test_df = pd.read_csv(test_csv)
    ds = TextDataset(test_df.text.values, None, tokenizer, 128)
    loader = DataLoader(ds, batch_size=16)

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"]
            mask = batch["attention_mask"]
            logits = model(ids, mask).cpu().numpy()
            all_logits.append(logits)

    logits = np.vstack(all_logits)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    out = pd.DataFrame({"id": test_df.id})
    for i, col in enumerate(LABEL_COLS):
        out[col] = preds[:, i]

    out.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

# -------------------------- ENSEMBLE COMBINE --------------------------
def ensemble_results(test_csv):
    print("\n🤝 Creating ENSEMBLE...")

    df1 = pd.read_csv("/kaggle/working/sub_bert-base-uncased.csv")
    df2 = pd.read_csv("/kaggle/working/sub_roberta-base.csv")
    df3 = pd.read_csv("/kaggle/working/sub_distilroberta-base.csv")

    final = df1.copy()
    for col in LABEL_COLS:
        final[col] = ((df1[col] + df2[col] + df3[col]) / 3 >= 0.5).astype(int)

    final.to_csv("/kaggle/working/submission.csv", index=False)
    print("Saved ENSEMBLE → submission.csv")

# -------------------------- MAIN --------------------------
def main(args):
    MODELS = ["bert-base-uncased", "roberta-base", "distilroberta-base"]

    for model in MODELS:
        train_model(model, args.train_csv, f"/kaggle/working/{model}-run", args.epochs)
        predict_model(model, f"/kaggle/working/{model}-run", args.test_csv,
                      f"/kaggle/working/sub_{model}.csv")

    ensemble_results(args.test_csv)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--output_dir", type=str, default="/kaggle/working/ensemble-run")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--run_name", type=str, default="ensemble-run")
    args = p.parse_args()

    # WandB Init
    if args.wandb_project and WANDB_OK:
        try:
            key = UserSecretsClient().get_secret("WANDB_API_KEY")
            wandb.login(key=key)
            wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        except Exception as e:
            print(f"⚠️ WandB Init Failed: {e}")

    main(args)

    if args.wandb_project and WANDB_OK:
        wandb.finish()
