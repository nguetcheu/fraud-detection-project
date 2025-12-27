import pandas as pd
import joblib
from src.models.shap_explain import explain_model

# Charger test set
test_df = pd.read_csv("data/processed/test.csv")
target_col = "is_fraud"
X_test = test_df.drop(columns=[target_col])

# Expliquer le modèle SMOTE
top_features = explain_model("results/models/smote.pkl", X_test, top_n=10)
print("Top 10 features les plus importantes selon SHAP :")
print(top_features)
