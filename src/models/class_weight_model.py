import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib

def train_rf_class_weight(X_train, y_train):
    """Random Forest avec class_weight='balanced'"""
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_gb_class_weight(X_train, y_train):
    """Gradient Boosting (sklearn n'a pas class_weight direct, mais on peut ajuster sample_weight)"""
    model = GradientBoostingClassifier(random_state=42)
    # Pour GB, on peut utiliser sample_weight si nécessaire (optionnel)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Calculer precision, recall, f1, roc_auc"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    return metrics

def save_model(model, path):
    """Sauvegarder le modèle"""
    joblib.dump(model, path)
