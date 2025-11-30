"""
Kaggle-ready prediction script.
Loads saved TF-IDF vectorizer + MLP model and generates submission.csv.

Usage example:
  python predict.py \
      --test_csv /kaggle/input/2025-sep-dl-gen-ai-project/test.csv \
      --model_dir /kaggle/working/models/scratch \
      --output_csv /kaggle/working/submission.csv
"""

import argparse
import pandas as pd
import joblib
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

LABEL_COLS = ["anger","fear","joy","sadness","surprise"]

# Dataset for TF-IDF features
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

# Simple MLP model
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

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load test CSV
    print(f"[INFO] Loading test CSV: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    texts = test_df[args.text_col].fillna("").astype(str).values

    # Load vectorizer
    vect_path = os.path.join(args.model_dir, "tfidf_vectorizer.joblib")
    print(f"[INFO] Loading vectorizer from: {vect_path}")
    vect = joblib.load(vect_path)
    X_test = vect.transform(texts)

    # Init model with correct input_dim
    input_dim = X_test.shape[1]
    model = MLP(input_dim=input_dim, output_dim=len(LABEL_COLS)).to(device)

    # Load trained weights
    model_path = os.path.join(args.model_dir, "scratch_model.pth")
    print(f"[INFO] Loading MLP model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prediction loader
    test_dataset = TfidfDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > args.threshold).cpu().numpy()
            all_preds.append(preds)

    preds_final = np.vstack(all_preds).astype(int)

    # Build submission df
    submission = pd.DataFrame({"id": test_df["id"]})
    for i, col in enumerate(LABEL_COLS):
        submission[col] = preds_final[:, i]

    print(f"[INFO] Saving final CSV to {args.output_csv}")
    submission.to_csv(args.output_csv, index=False)
    print("[INFO] Prediction complete. Submission CSV created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="/kaggle/input/2025-sep-dl-gen-ai-project/test.csv")
    parser.add_argument("--model_dir", type=str, default="/kaggle/working/models/scratch")
    parser.add_argument("--output_csv", type=str, default="/kaggle/working/submission.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    main(args)
