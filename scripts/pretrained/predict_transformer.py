import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn as nn

LABEL_COLS=["anger","fear","joy","sadness","surprise"]

class PredDataset(Dataset):
    def __init__(self,texts,tokenizer,max_len):
        self.texts=texts
        self.tokenizer=tokenizer
        self.max_len=max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self,idx):
        enc=self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {k:v.squeeze(0) for k,v in enc.items()}

class GenericTransformer(nn.Module):
    def __init__(self, model_dir, num_labels, dropout=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, config=self.config, local_files_only=True)
        h = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h, num_labels)

    def forward(self, ids, mask):
        out = self.model(ids, mask)
        pooled = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))

def main(args):
    device="cuda" if torch.cuda.is_available() else "cpu"

    tok=AutoTokenizer.from_pretrained(args.local_model_path, local_files_only=True)

    model=GenericTransformer(args.local_model_path, len(LABEL_COLS))
    model.load_state_dict(torch.load(f"{args.model_dir}/best_model.pt", map_location=device))
    model.to(device)

    df=pd.read_csv(args.test_csv)
    texts=df["text"].astype(str).values

    ds=PredDataset(texts,tok,args.max_len)
    dl=DataLoader(ds,batch_size=args.batch_size)

    model.eval()
    outs=[]
    with torch.no_grad():
        for b in dl:
            ids=b["input_ids"].to(device)
            mask=b["attention_mask"].to(device)
            logits=model(ids,mask).cpu().numpy()
            outs.append(logits)

    logits=np.vstack(outs)
    probs=1/(1+np.exp(-logits))
    preds=(probs>=args.threshold).astype(int)

    sub=pd.DataFrame({"id":df["id"]})
    for i,c in enumerate(LABEL_COLS):
        sub[c]=preds[:,i]

    sub.to_csv(args.out,index=False)
    print("Saved:",args.out)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--test_csv",type=str)
    p.add_argument("--model_dir",type=str)
    p.add_argument("--local_model_path",type=str)
    p.add_argument("--out",type=str)
    p.add_argument("--batch_size",type=int,default=16)
    p.add_argument("--max_len",type=int,default=128)
    p.add_argument("--threshold",type=float,default=0.5)
    args=p.parse_args()
    main(args)
