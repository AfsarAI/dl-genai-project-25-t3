# Kaggle-ready train.py (TF-IDF + MLP) with FULL WandB Logging

import argparse
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score  # Added accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# W&B for Kaggle
try:
    import wandb
    from kaggle_secrets import UserSecretsClient
    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False

LABEL_COLS = ["anger","fear","joy","sadness","surprise"]

# Dataset
class TfidfDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.toarray(), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x

def load_data(path):
    return pd.read_csv(path)

# -------------- W&B Setup for Kaggle --------------
def setup_wandb(project, run_name, args_dict):
    user_secrets = UserSecretsClient()
    key = None
    try:
        key = user_secrets.get_secret("WANDB_API_KEY")
    except:
        pass

    if key:
        wandb.login(key=key)
        # reinit=True allows multiple runs in same session if needed
        run = wandb.init(project=project, name=run_name, config=args_dict, reinit=True)
        return run
    else:
        os.environ["WANDB_MODE"] = "offline"
        run = wandb.init(project=project, name=run_name, config=args_dict, reinit=True)
        return run

# ---------------------------------------------------

def build_and_train(X_train_texts, X_val_texts, y_train, y_val, max_features, epochs, batch_size, lr, output_dir, use_wandb):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vect = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    Xtr = vect.fit_transform(X_train_texts)
    Xv = vect.transform(X_val_texts)

    train_loader = DataLoader(TfidfDataset(Xtr, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TfidfDataset(Xv, y_val), batch_size=batch_size, shuffle=False)

    model = MLP(input_dim=Xtr.shape[1], output_dim=len(LABEL_COLS)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_f1 = -1

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop (Now calculates Loss + Accuracy + F1)
        model.eval()
        val_loss = 0
        preds_list, labels_list = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                
                # Calculate val loss
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                # Predictions
                pred = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                preds_list.append(pred)
                labels_list.append(y_batch.cpu().numpy())

        preds_final = np.vstack(preds_list)
        labels_final = np.vstack(labels_list)
        
        avg_val_loss = val_loss / len(val_loader)
        macro_f1 = f1_score(labels_final, preds_final, average='macro')
        acc = accuracy_score(labels_final.flatten(), preds_final.flatten()) # Flatten for overall accuracy

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {macro_f1:.4f} | Val Acc: {acc:.4f}")

        # --- LOGGING TO WANDB ---
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": acc,
                "val_f1": macro_f1
            })

        # Save best model
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "scratch_model.pth"))

    return vect, model, best_f1

def save_artifacts(vect, outdir):
    os.makedirs(outdir, exist_ok=True)
    joblib.dump(vect, os.path.join(outdir, "tfidf_vectorizer.joblib"))

def main(args):
    if args.use_wandb and WANDB_AVAILABLE:
        setup_wandb(args.wandb_project, args.run_name, vars(args))

    df = load_data(args.train_csv)
    texts = df[args.text_col].fillna("").astype(str).values
    y = df[LABEL_COLS].values

    X_train_texts, X_val_texts, y_train, y_val = train_test_split(
        texts, y, test_size=args.val_size, random_state=42, shuffle=True
    )

    vect, model, best_f1 = build_and_train(
        X_train_texts, X_val_texts, y_train, y_val,
        args.max_features, args.epochs, args.batch_size,
        args.lr, args.output_dir, args.use_wandb
    )

    print("[INFO] Best Val Macro F1:", best_f1)
    save_artifacts(vect, args.output_dir)

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="/kaggle/input/2025-sep-dl-gen-ai-project/train.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/models/scratch")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--max_features", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dl-genai-kaggle")
    parser.add_argument("--run_name", type=str, default="tfidf-scratch-kaggle")
    args = parser.parse_args()
    main(args)
