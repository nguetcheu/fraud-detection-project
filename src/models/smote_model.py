import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def train_smote_model(X_train, y_train):
    """Appliquer SMOTE sur le train set puis entraîner Random Forest"""
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_res, y_res)
    return model

def evaluate_model(model, X_test, y_test):
    """Calculer métriques sur test set"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    return metrics

def save_model(model, path="results/models/smote.pkl"):
    joblib.dump(model, path)
