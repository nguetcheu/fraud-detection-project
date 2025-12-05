import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_dataset(path: str = "data/raw/transaction.csv") -> pd.DataFrame:
  """Charger le dataset brut depuis le dossier data/raw"""
  return pd.read_csv(path)

def save_processed_data(df: pd.DataFrame, path: str = "data/processed/dataset_clean.csv"):
    """Sauvegarder le dataset nettoyé dans data/processed"""
    df.to_csv(path, index=False)

def show_data_info(df: pd.DataFrame):
  """Afficher les informations principales du dataset"""
  print("DataFrame Info:")
  print(df.info())
  print("\nMissing Values per Column:")
  print(df.isnull().sum())
  print("\nStatistical Summary:")
  print(df.describe(include='all'))
  
def train_test_split_data(df):
  """Diviser le dataset en ensembles d'entraînement et de test"""
  train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
  return train_df, test_df