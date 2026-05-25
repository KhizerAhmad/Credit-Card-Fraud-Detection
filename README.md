# Credit Card Fraud Detection 💳

An ML system that detects fraudulent credit card transactions in a highly imbalanced dataset of 280k+ records. The key challenge here isn't just building a model — it's understanding why accuracy is a useless metric when 99.8% of your data is non-fraud, and evaluating with F1-Score and confusion matrices instead.

---

## What it does

- Loads and preprocesses 280,000+ credit card transaction records
- Handles class imbalance with stratified train/test splits
- Scales features using StandardScaler
- Trains and compares 4 ML models
- Evaluates using F1-Score and confusion matrices — not just accuracy
- Saves all evaluation plots to the `Graphs/` folder

---

## The Imbalance Problem

This is what makes fraud detection genuinely hard:

| Class | Count | Percentage |
|-------|-------|------------|
| Legitimate transactions | ~284,000 | ~99.83% |
| Fraudulent transactions | ~492 | ~0.17% |

A model that just predicts "not fraud" every time gets **99.8% accuracy** but catches **zero fraud**. That's why this project evaluates with **F1-Score** — which balances precision and recall — and uses **stratified splits** to ensure fraud cases appear proportionally in both train and test sets.

---

## Models Compared

| Model | Notes |
|-------|-------|
| K-Nearest Neighbors (KNN) | Distance-based, sensitive to scale |
| Logistic Regression | Linear baseline, works well with scaled data |
| Decision Tree | Tree-based, prone to overfitting on imbalanced data |
| Random Forest | Ensemble method, generally more robust |

---

## Why I built this

Fraud detection is one of those problems where the data itself is the challenge — not the model. Wanted to understand class imbalance properly, learn why naive accuracy fails, and practice using the right evaluation metrics. It's a much more realistic ML workflow than clean, balanced datasets.

---

## Tech Stack

| Library | Usage |
|---------|-------|
| Python | Core language |
| scikit-learn | Models, StandardScaler, stratified split, metrics |
| pandas | Data loading and manipulation |
| NumPy | Array operations |
| matplotlib / seaborn | Confusion matrices and evaluation plots |
| UV | Package manager (`uv.lock` for reproducible installs) |

---

## How to run it

**1. Clone the repo**
```bash
git clone https://github.com/KhizerAhmad/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

**2. Install dependencies**

With pip:
```bash
pip install -r requirements.txt
```

Or with UV:
```bash
uv sync
```

**3. Add the dataset**

Download the Credit Card Fraud Detection dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project directory.

**4. Run it**
```bash
python main.py
```

---

## Project Structure

```
Credit-Card-Fraud-Detection/
│
├── main.py              # Full pipeline — preprocessing, training, evaluation
├── Graphs/              # Confusion matrices and evaluation plots
├── requirements.txt     # pip dependencies
├── pyproject.toml       # Project metadata
├── uv.lock              # UV lockfile
└── .gitignore
```

---

## Pipeline Overview

```
Raw Data (280k+ records) 
  → StandardScaler (feature scaling)
  → Stratified Train/Test Split (preserve fraud ratio)
       → KNN               ┐
       → Logistic Reg      ├─ Evaluate via F1-Score + Confusion Matrix
       → Decision Tree     │
       → Random Forest     ┘
            → Compare → Save Graphs
```

---

## Evaluation Metrics Used

- **F1-Score** — primary metric (balances precision and recall)
- **Confusion Matrix** — shows true positives, false positives, false negatives
- **Classification Report** — per-class precision, recall, F1
- Accuracy intentionally deprioritized due to class imbalance

---

## Screenshots

<img width="796" height="564" alt="Accuracy Graph" src="https://github.com/user-attachments/assets/9516a05c-47bc-45b3-86b6-d09cad00ad45" />
<img width="796" height="564" alt="Class Distribution" src="https://github.com/user-attachments/assets/a91d94ca-929d-414d-91a6-9e31c628f304" />


---

## Author

**Khizer Ahmad** — built this to understand real-world ML challenges: class imbalance, proper metric selection, and why a high accuracy score can be completely meaningless.

Feel free to fork it and try adding SMOTE oversampling or an anomaly detection approach.
