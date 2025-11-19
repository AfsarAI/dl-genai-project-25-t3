# scripts/train.py
"""
Simple baseline trainer: TF-IDF + LogisticRegression (multioutput).
Usage:
  python scripts/train.py --train_csv data/raw/train.csv --output_dir models --wandb_project 23f2002023-t32025
"""
import argparse
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier # Import MultiOutputClassifier
import warnings
warnings.filterwarnings("ignore")

# Optional: import wandb if you want logging. If not installed, skip.
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

LABEL_COLS = ["anger","fear","joy","sadness","surprise"]

def load_data(path):
    df = pd.read_csv(path)
    # ensure label columns exist (if dataset different, adapt this)
    return df

def prepare_texts(df, text_col="text"):
    return df[text_col].fillna("").astype(str).values

def prepare_targets(df):
    return df[LABEL_COLS].fillna(0).astype(int).values

def build_and_train(X_train_texts, X_val_texts, y_train, y_val, max_features=20000):
    vect = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    Xtr = vect.fit_transform(X_train_texts)
    Xv   = vect.transform(X_val_texts)
    # Wrap LogisticRegression with MultiOutputClassifier
    base_clf = LogisticRegression(max_iter=1000, solver='liblinear') # Added solver for clarity
    clf = MultiOutputClassifier(base_clf)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xv)
    macro_f1 = f1_score(y_val, preds, average='macro')
    return vect, clf, macro_f1

def save_artifacts(vect, clf, outdir):
    os.makedirs(outdir, exist_ok=True)
    joblib.dump(vect, os.path.join(outdir, "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(outdir, "logreg_model.joblib"))

def main(args):
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    df = load_data(args.train_csv)
    # split train/val
    texts = prepare_texts(df, args.text_col)
    y = prepare_targets(df)
    X_train_texts, X_val_texts, y_train, y_val = train_test_split(
        texts, y, test_size=args.val_size, random_state=42, shuffle=True
    )

    vect, clf, macro_f1 = build_and_train(X_train_texts, X_val_texts, y_train, y_val,
                                         max_features=args.max_features)

    print(f"[INFO] Val Macro F1: {macro_f1:.4f}")
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({"val_macro_f1": macro_f1})

    save_artifacts(vect, clf, args.output_dir)
    print(f"[INFO] Saved artifacts to {args.output_dir}")

    if args.use_wandb and WANDB_AVAILABLE:
        artifact = wandb.Artifact('baseline-logreg', type='model')
        artifact.add_file(os.path.join(args.output_dir, "logreg_model.joblib"))
        wandb.log_artifact(artifact)
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="data/raw/train.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--max_features", type=int, default=20000)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="23f2002023-t32025")
    parser.add_argument("--run_name", type=str, default="tfidf-logreg")
    args = parser.parse_args()
    main(args)
