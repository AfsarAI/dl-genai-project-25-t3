# scripts/predict.py
"""
Load saved TF-IDF + LR artifacts and predict on test CSV to create submission.csv.
Usage:
  python scripts/predict.py --test_csv data/raw/test.csv --model_dir models --output_csv submission.csv
"""
import argparse
import pandas as pd
import joblib
import os

LABEL_COLS = ["anger","fear","joy","sadness","surprise"]

def main(args):
    test = pd.read_csv(args.test_csv)
    vect = joblib.load(os.path.join(args.model_dir, "tfidf_vectorizer.joblib"))
    clf = joblib.load(os.path.join(args.model_dir, "logreg_model.joblib"))
    texts = test[args.text_col].fillna("").astype(str).values
    Xtest = vect.transform(texts)
    preds = clf.predict(Xtest).astype(int)  # shape (n,5)
    submission = pd.DataFrame({"id": test["id"]})
    for i, col in enumerate(LABEL_COLS):
        submission[col] = preds[:, i]
    submission.to_csv(args.output_csv, index=False)
    print(f"[INFO] Saved submission to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="data/raw/test.csv")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--output_csv", type=str, default="submission.csv")
    parser.add_argument("--text_col", type=str, default="text")
    args = parser.parse_args()
    main(args)
