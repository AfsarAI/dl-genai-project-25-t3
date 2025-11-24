
# DL-GenAI Baseline (23f2002023)

## What's included
- scripts/train_transformer_multilabel.py   -> TF-IDF + LogisticRegression baseline trainer
- scripts/predict_transformer.py -> Create submission.csv from saved model
- requirements.txt

## How to run (example)
1. Install dependencies:
   `pip install -r requirements.txt`

2. Train:
   `python scripts/train_transformer_multilabel.py --train_csv data/raw/train.csv --output_dir models`

3. Predict:
   `python scripts/predict_transformer.py --test_csv data/raw/test.csv --model_dir models --output_csv submission.csv`
