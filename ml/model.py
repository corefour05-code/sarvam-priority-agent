"""
Calibrated Machine Learning Model for Repayment & Reachability Prediction.
Uses regularized, calibrated Logistic Regression for transparent, auditable probability estimation.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from ml.features import extract_features_dataframe, extract_features_from_dict, FEATURE_NAMES
from config import settings

class RepaymentPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.raw_lr = None
        self.is_trained = False
        
    def train(self, df: pd.DataFrame):
        """
        Trains calibrated logistic regression on historical/synthetic dataset.
        """
        X = extract_features_dataframe(df)
        y_pay = df["true_will_pay"].astype(int).values
        
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Train baseline interpretable Logistic Regression
        self.raw_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.raw_lr.fit(X_scaled, y_pay)
        
        # Train calibrated classifier to output genuine Bayesian probabilities
        self.model = CalibratedClassifierCV(estimator=self.raw_lr, method="sigmoid", cv=5)
        self.model.fit(X_scaled, y_pay)
        
        self.is_trained = True
        
        # Evaluate model quality metrics
        preds_proba = self.model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y_pay, preds_proba)
        brier = brier_score_loss(y_pay, preds_proba)
        print(f"[ML Model Trained] ROC-AUC: {auc:.4f} | Brier Score (Calibration): {brier:.4f}")
        
    def save(self, path: str = None):
        if path is None:
            path = settings.MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "model": self.model,
            "raw_lr": self.raw_lr,
            "feature_names": FEATURE_NAMES
        }, path)
        print(f"Model saved to {path}")
        
    def load(self, path: str = None):
        if path is None:
            path = settings.MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found. Train first.")
        bundle = joblib.load(path)
        self.scaler = bundle["scaler"]
        self.model = bundle["model"]
        self.raw_lr = bundle["raw_lr"]
        self.is_trained = True
        
    def predict_repayment_probability(self, loan_dict: dict) -> float:
        """
        Predicts calibrated probability that borrower will pay if contacted today.
        """
        if not self.is_trained:
            self.load()
            
        features = extract_features_from_dict(loan_dict)
        # Fast numpy standardization
        features_scaled = (features - self.scaler.mean_) / self.scaler.scale_
        prob = self.model.predict_proba(features_scaled.reshape(1, -1))[0, 1]
        return float(np.clip(prob, 0.01, 0.99))
        
    def predict_batch_repayment_probabilities(self, loan_dicts: list) -> np.ndarray:
        """
        Fast batch prediction across thousands of loans using vectorization.
        """
        if not self.is_trained:
            self.load()
        if not loan_dicts:
            return np.array([])
            
        feature_matrix = np.array([extract_features_from_dict(d) for d in loan_dicts], dtype=np.float32)
        scaled_matrix = (feature_matrix - self.scaler.mean_) / self.scaler.scale_
        probs = self.model.predict_proba(scaled_matrix)[:, 1]
        return np.clip(probs, 0.01, 0.99)
        
    def predict_contact_probability(self, loan_dict: dict) -> float:
        """
        Predicts probability that borrower will answer the call based on historical reachability.
        """
        attempts = float(loan_dict.get("contact_attempts_count", 0) or 0)
        responses = float(loan_dict.get("contact_responses_count", 0) or 0)
        
        if attempts == 0:
            return 0.70 # Prior for uncalled borrowers
            
        response_rate = responses / attempts
        # Apply slight decay if too many failed attempts (burnout)
        burnout_factor = max(0.4, 1.0 - (attempts * 0.05))
        prob = (0.25 + 0.70 * response_rate) * burnout_factor
        return float(np.clip(prob, 0.10, 0.95))

predictor = RepaymentPredictor()

def train_default_model():
    data_path = settings.SYNTHETIC_DATA_PATH
    if not os.path.exists(data_path):
        from database.seed_data import generate_dataset
        generate_dataset(4200)
        
    df = pd.read_csv(data_path)
    pred = RepaymentPredictor()
    pred.train(df)
    pred.save()
    return pred

if __name__ == "__main__":
    train_default_model()
