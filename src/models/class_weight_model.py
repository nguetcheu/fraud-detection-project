import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

def train_rf_class_weight(X_train, y_train):
    """Random Forest avec class_weight='balanced'"""
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_gb_class_weight(X_train, y_train):
    """Gradient Boosting avec pondération des classes via sample_weight"""
    model = GradientBoostingClassifier(random_state=42)
    # Calcul des poids pour chaque échantillon
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model

def evaluate_model(model, X_test, y_test):
    """Calculer precision, recall, f1, roc_auc"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
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
