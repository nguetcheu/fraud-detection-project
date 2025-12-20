import pandas as pd
import matplotlib.pyplot as plt
from src.data.load_data import load_dataset, show_data_info
from src.data.preprocess import clean_data

# 1. Charger le dataset
df = load_dataset("data/raw/transactions.csv")

# 3. Afficher informations principales
show_data_info(df)

