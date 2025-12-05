import os
import pandas as pd
from src.data.preprocess import clean_data
from src.data.load_data import load_dataset, train_test_split_data, save_processed_data
from src.models.baseline_model import train_baseline_model, evaluate_model, save_model
import json

# Charger et nettoyer les données
df = load_dataset("data/raw/transactions.csv")
df_clean = clean_data(df)

# Split train/test si pas déjà fait
train_df, test_df = train_test_split_data(df_clean)
save_processed_data(train_df, "data/processed/train.csv")
save_processed_data(test_df, "data/processed/test.csv")

# Séparer features / target
target_col = "is_fraud"
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# Entraîner le modèle baseline
model = train_baseline_model(X_train, y_train)

# Évaluer le modèle
metrics = evaluate_model(model, X_test, y_test)
print("Metrics baseline:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Sauvegarder le modèle et les métriques
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)
save_model(model, "results/models/baseline.pkl")

with open("results/metrics/baseline.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Baseline terminé. Modèle et métriques sauvegardés.")
