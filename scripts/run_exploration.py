import pandas as pd
import matplotlib.pyplot as plt
from src.data.load_data import load_dataset, show_data_info
from src.data.preprocess import clean_data

# 1. Charger le dataset
df = load_dataset("data/raw/transactions.csv")

# 2. Nettoyer les données
df_clean = clean_data(df)

# 3. Afficher informations principales
show_data_info(df_clean)

# 4. Visualiser distribution des classes
plt.figure(figsize=(6,6))
df_clean['is_fraud'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Distribution des classes (Fraude vs Légitime)")
plt.savefig("results/figures/class_distribution_pie.png")
plt.close()

plt.figure(figsize=(8,5))
df_clean['is_fraud'].value_counts().plot.bar()
plt.title("Histogramme des classes")
plt.xlabel("Classe")
plt.ylabel("Nombre de transactions")
plt.savefig("results/figures/class_distribution_bar.png")
plt.close()