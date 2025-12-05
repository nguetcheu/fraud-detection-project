import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
  """
  Nettoyer et préparer les données pour ML :
  - Remplacer les valeurs manquantes numériques par la médiane
  - Arrondir certaines colonnes float à 2 décimales
  - Supprimer .0 pour les colonnes entières
  - Encoder les colonnes catégorielles pour ML
  - Transformer date_transaction en année et mois
  """
  df = df.copy()
    
  # Remplacer les valeurs manquantes
  for col in df.columns:
      if df[col].dtype in ['float64', 'int64', 'Int64']:
        median = df[col].median()
        df[col] = df[col].fillna(median)
      else:
        df[col] = df[col].fillna("Inconnu")
  
  # Arrondir à 2 décimales pour certaines colonnes float
  round_2_cols = ["anciennete_compte", "revenus_mensuels", "montant_transaction", "montant_total_24h", "distance_trans_precedente", "temps_depuis_derniere_trans", "montant_moyen_30j", "ration_montant_moyen"]
  for col in round_2_cols:
      if col in df.columns:
        df[col] = df[col].round(2)
        
  # Colonnes entières (supprimer .0)
  int_cols = ["nb_chambres", "annee_construction", "parking", "score_commerces"]
  for col in int_cols:
    if col in df.columns:
      df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else 0)
      df[col] = df[col].astype("Int64")
      
  # Encoder les colonnes catégorielles
    cat_cols = ["type_compte", "statut_professionnel", "region", "type_transaction", "categorie_marchand", "pays_transaction", "mode_paiement"]
    for col in cat_cols:
      if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
      
  # Transformer date_transaction en année et mois
  if "date_transaction" in df.columns:
      df["date_transaction"] = pd.to_datetime(df["date_transaction"], errors='coerce')
      df["annee_transaction"] = df["date_transaction"].dt.year.fillna(df["date_transaction"].dt.year.median())
      df["mois_transaction"] = df["date_transaction"].dt.month.fillna(df["date_transaction"].dt.month.median())
      df = df.drop(columns=["date_transaction"])
    
  return df