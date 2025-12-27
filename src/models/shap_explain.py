import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

def explain_model(model_path: str, X_test: pd.DataFrame, top_n: int = 10):
  """
  Générer les SHAP values et les plots pour un modèle binaire.
    
  Args:
    model_path (str): chemin vers le modèle sauvegardé (.pkl)
    X_test (pd.DataFrame): données de test
    top_n (int): nombre de features les plus importantes à retourner
        
  Returns:
    list: noms des top_n features les plus importantes pour la classe 1
  """
  # Charger modèle
  model = joblib.load(model_path)
    
  # Explainer Tree pour RandomForest ou GradientBoosting
  explainer = shap.TreeExplainer(model)
    
  # Calcul SHAP values
  shap_values = explainer.shap_values(X_test)
    
  # Pour modèle binaire, shap_values est une liste ou un array 3D
  if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Classe 1 = fraude
  elif shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]  # Classe 1
    
  # Vérification des dimensions
  assert shap_values.shape == X_test.shape, \
    f"Shape mismatch: shap_values {shap_values.shape}, X_test {X_test.shape}"
    
  # Summary plot (bar)
  os.makedirs("results/figures", exist_ok=True)
  shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
  plt.savefig("results/figures/shap_summary_bar.png")
  plt.close()
    
  # Summary plot (beeswarm)
  shap.summary_plot(shap_values, X_test, show=False)
  plt.savefig("results/figures/shap_summary_beeswarm.png")
  plt.close()
    
  # Top features
  mean_abs_shap = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": abs(shap_values).mean(axis=0)
  }).sort_values(by="mean_abs_shap", ascending=False)
    
  top_features = mean_abs_shap.head(top_n)["feature"].tolist()
    
  print(f"Top {top_n} features pour la fraude : {top_features}")
    
  return top_features
