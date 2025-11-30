import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel
import os

LABEL_COLS = ["anger","fear","joy","sadness","surprise"]

LOCAL_MODEL = "/kaggle/working/roberta-base-local"   # <---- FIX

class TextDatasetPred(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

class TransformerForMultiLabelInfer(torch.nn.Module):
    def __init__(self, model_dir, n_labels, dropout=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.backbone = AutoModel.from_pretrained(model_dir, config=self.config, local_files_only=True)

        h = self.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(h, n_labels)

    def forward(self, ids, mask):
        out = self.backbone(input_ids=ids, attention_mask=mask)
        last = out.last_hidden_state
        m = mask.unsqueeze(2)
        pooled = (last * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.classifier(self.dropout(pooled))

def predict(loader, model, device):
    model.eval()
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, mask).cpu().numpy()
            all_logits.append(logits)

    return np.vstack(all_logits)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL, local_files_only=True)

    model = TransformerForMultiLabelInfer(
        LOCAL_MODEL,
        n_labels=len(LABEL_COLS),
        dropout=0.1
    )
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "best_model.pt"), map_location=device))
    model.to(device)

    test_df = pd.read_csv(args.test_csv)
    texts = test_df["text"].astype(str).values

    ds = TextDatasetPred(texts, tokenizer, args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    logits = predict(loader, model, device)
    probs = 1 / (1 + np.exp(-logits))

    preds = (probs >= args.threshold).astype(int)

    sub = pd.DataFrame({"id": test_df["id"]})
    for i, col in enumerate(LABEL_COLS):
        sub[col] = preds[:, i]

    sub.to_csv(args.out, index=False)
    print("Saved:", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv", type=str)
    p.add_argument("--model_dir", type=str)
    p.add_argument("--out", type=str)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    main(args)
