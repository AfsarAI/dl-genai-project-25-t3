
# DL-GenAI Project Baseline (23f2002023)

## Project Overview & Level 1 Viva Checkpoints

This project aims to develop and compare multiple deep learning models for multi-label text classification. The structure and development adhere to the following Level 1 Viva checkpoints:

1.  **Model Completion**: At least three unique models are implemented. Currently, this includes:
    *   A model built **from scratch** (TF-IDF + MLP).
    *   A **pretrained** transformer model (e.g., RoBERTa, DeBERTa).
    *   (Future: A third model of choice).

2.  **Wandb Tracking**: All model training runs are tracked using Weights & Biases (W&B). This allows for easy comparison of model performance (e.g., F1 score, accuracy) across different experiments.

## What's Included
-   `scripts/scratch/train.py`: Script to train the TF-IDF + "from scratch" MLP model.
-   `scripts/scratch/predict.py`: Script to generate predictions using the TF-IDF + "from scratch" MLP model.
-   `scripts/pretrained/train_transformer_multilabel.py`: Script to train a pretrained transformer model (e.g., RoBERTa).
-   `scripts/pretrained/predict_transformer.py`: Script to generate predictions using a pretrained transformer model (e.g., RoBERTa).
-   `scripts/extra/`: Directory for a third model of choice (currently empty).
-   `requirements.txt`: List of Python dependencies.
-   `.gitignore`: Specifies files/directories to be ignored by Git.

## How to Run (Example for Scratch MLP)

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the scratch MLP model** (with W&B tracking):
    ```bash
    python scripts/scratch/train.py --train_csv data/raw/train.csv --output_dir models/scratch --use_wandb --wandb_project 23f2002023-dl-genai-project --run_name "tfidf-mlp-scratch"
    ```

3.  **Predict using the scratch MLP model**:
    ```bash
    python scripts/scratch/predict.py --test_csv data/raw/test.csv --model_dir models/scratch --output_csv submission_scratch_mlp.csv
    ```

4.  **Submit to Kaggle** (after generating `submission_scratch_mlp.csv`):
    ```bash
    kaggle competitions submit -c 2025-sep-dl-gen-ai-project -f submission_scratch_mlp.csv -m "TF-IDF + Scratch MLP baseline"
    ```

## How to Run (Example for Pretrained RoBERTa)

1.  **Prepare validation split** (if not already done):
    ```bash
    # Run the cell that creates data/raw/val.csv from train.csv
    # (e.g., cell BcS4pOFVcusW or wmckZLovipv1 in the notebook if using optimized scripts)
    ```

2.  **Train a pretrained transformer model** (e.g., `roberta-base` with W&B tracking):
    ```bash
    python scripts/pretrained/train_transformer_multilabel.py \
      --train_csv data/raw/train.csv \
      --val_frac 0.1 \
      --model_name roberta-base \
      --output_dir models/roberta-base \
      --epochs 3 \
      --batch_size 8 \
      --lr 2e-5 \
      --wandb_project Main-Trainings \
      --run_name roberta-run1
    ```

3.  **Predict using the pretrained model** (e.g., `roberta-base`):
    ```bash
    python scripts/pretrained/predict_transformer.py \
      --test_csv data/raw/test.csv \
      --val_csv data/raw/val.csv \
      --model_dir models/roberta-base \
      --model_name roberta-base \
      --tune_threshold \
      --out submission_roberta.csv
    ```

4.  **Submit to Kaggle**:
    ```bash
    kaggle competitions submit \
      -c 2025-sep-dl-gen-ai-project \
      -f submission_roberta.csv \
      -m "Roberta model Run-1 submission"
    ```
