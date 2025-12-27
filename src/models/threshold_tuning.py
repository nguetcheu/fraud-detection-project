import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np
import os

def find_optimal_threshold(y_true, y_probs):
  """Calculer précision, recall, F1 pour différents seuils et trouver le seuil optimal"""
  precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
  # Calculer F1 pour chaque seuil
  f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
  # Trouver le seuil correspondant au F1 maximal
  idx = np.argmax(f1_scores)
  best_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
  return best_threshold, precisions, recalls, f1_scores, thresholds

def plot_f1_vs_threshold(f1_scores, thresholds, save_path="results/figures/f1_vs_threshold.png"):
  """Tracer F1-score en fonction du seuil"""
  plt.figure(figsize=(8,5))
  plt.plot(thresholds, f1_scores[:-1], marker='o')
  plt.xlabel("Seuil")
  plt.ylabel("F1-score")
  plt.title("F1-score vs Threshold")
  plt.grid(True)
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  plt.savefig(save_path)
  plt.close()