import os
import pandas as pd
import json
from src.data.preprocess import clean_data
from src.data.load_data import load_dataset, train_test_split_data, save_processed_data
from src.models.class_weight_model import train_rf_class_weight, train_gb_class_weight, evaluate_model, save_model

# Charger train/test
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# Séparer features / target
target_col = "is_fraud"
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# Entraîner Random Forest class_weight
rf_model = train_rf_class_weight(X_train, y_train)
rf_metrics = evaluate_model(rf_model, X_test, y_test)

# Entraîner Gradient Boosting class_weight
gb_model = train_gb_class_weight(X_train, y_train)
gb_metrics = evaluate_model(gb_model, X_test, y_test)

# Sauvegarder modèles et métriques
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

save_model(rf_model, "results/models/rf_class_weight.pkl")
save_model(gb_model, "results/models/gb_class_weight.pkl")

with open("results/metrics/rf_class_weight.json", "w") as f:
    json.dump(rf_metrics, f, indent=4)
with open("results/metrics/gb_class_weight.json", "w") as f:
    json.dump(gb_metrics, f, indent=4)

print("Modèles de pondération des classes entraînés et enregistrés.")
