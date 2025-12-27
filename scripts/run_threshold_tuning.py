import os
import pandas as pd
import joblib
from src.data.load_data import load_dataset
from src.models.threshold_tuning import find_optimal_threshold, plot_f1_vs_threshold
from src.models.smote_model import evaluate_model

# Charger test set
test_df = pd.read_csv("data/processed/test.csv")
target_col = "is_fraud"
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# Charger le meilleur modèle (SMOTE par exemple)
model = joblib.load("results/models/smote.pkl")

# Prédictions probabilistes
y_probs = model.predict_proba(X_test)[:,1]

# Trouver seuil optimal
best_threshold, precisions, recalls, f1_scores, thresholds = find_optimal_threshold(y_test, y_probs)
print(f"Seuil optimal basé sur F1-score: {best_threshold:.2f}")

# Tracer F1 vs seuil
os.makedirs("results/figures", exist_ok=True)
plot_f1_vs_threshold(f1_scores, thresholds)

# Évaluer performance au seuil optimal
y_pred_opt = (y_probs >= best_threshold).astype(int)
metrics_opt = {
  "precision": (y_pred_opt & y_test).sum() / max(y_pred_opt.sum(),1),
  "recall": (y_pred_opt & y_test).sum() / y_test.sum(),
}
metrics_opt["f1"] = 2 * (metrics_opt["precision"] * metrics_opt["recall"]) / (metrics_opt["precision"] + metrics_opt["recall"] + 1e-8)

print("Metrics au seuil optimal:")
for k, v in metrics_opt.items():
  print(f"{k}: {v:.4f}")
