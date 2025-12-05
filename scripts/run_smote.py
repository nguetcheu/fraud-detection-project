import os
import pandas as pd
import json
from src.models.smote_model import train_smote_model, evaluate_model, save_model

# Charger train/test
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

target_col = "is_fraud"
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# Entraîner modèle avec SMOTE
model = train_smote_model(X_train, y_train)

# Évaluer
metrics = evaluate_model(model, X_test, y_test)
print("Metrics SMOTE:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Sauvegarder modèle et métriques
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

save_model(model, "results/models/smote.pkl")
with open("results/metrics/smote.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Modèle SMOTE entraîné et enregistré.")
