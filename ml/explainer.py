"""
Manager & Regulator Explainability Layer.
Generates human-readable, auditable factor attributions for every borrower score.
Directly satisfies the compliance red line: 'No score without a reason a manager could read aloud.'
"""

from typing import List, Dict, Any
import numpy as np
from ml.features import extract_features_from_dict, FEATURE_NAMES

HUMAN_FACTOR_DESCRIPTIONS = {
    "promise_fulfillment_ratio": {
        "positive": "Strong history of keeping payment commitments",
        "negative": "Low commitment fulfillment rate in past cycles"
    },
    "has_prior_promises": {
        "positive": "Borrower has active track record of making commitments",
        "negative": "No prior promise history on record"
    },
    "installment_repay_ratio": {
        "positive": "Consistently paid substantial portion of total installments",
        "negative": "Low overall installment completion track record"
    },
    "dpd_normalized": {
        "positive": "Early delinquency stage (low days overdue)",
        "negative": "High days past due (aging debt bucket)"
    },
    "overdue_to_principal_ratio": {
        "positive": "Manageable overdue balance relative to principal",
        "negative": "Severe overdue balance burden"
    },
    "recency_days_normalized": {
        "positive": "Recent payment activity recorded",
        "negative": "Extended dormancy with no recent payments"
    },
    "contact_response_rate": {
        "positive": "Responsive to past outreach reminders",
        "negative": "Poor responsiveness to previous call attempts"
    },
    "broken_promise_penalty": {
        "positive": "No recent broken promises",
        "negative": "Recent promise was breached without payment"
    },
    "ptp_due_today_signal": {
        "positive": "Committed promise-to-pay is due today",
        "negative": "No scheduled commitment due today"
    }
}

class ScoreExplainer:
    def __init__(self, predictor):
        self.predictor = predictor
        
    def explain(self, loan_dict: dict, prob_pay: float, prob_contact: float, expected_recovery: float, recommended_action: str) -> Dict[str, Any]:
        """
        Calculates local feature attributions using the logistic regression weights.
        """
        features = extract_features_from_dict(loan_dict)
        scaler = self.predictor.scaler
        raw_lr = self.predictor.raw_lr
        
        # Calculate standardized feature contributions: weight * (x - mean) / scale
        scaled_features = (features - scaler.mean_) / scaler.scale_
        weights = raw_lr.coef_[0]
        contributions = weights * scaled_features
        
        factors = []
        for i, name in enumerate(FEATURE_NAMES):
            contrib = float(contributions[i])
            if abs(contrib) < 0.10: # Skip negligible factors
                continue
                
            impact = "positive" if contrib > 0 else "negative"
            desc = HUMAN_FACTOR_DESCRIPTIONS.get(name, {}).get(impact, name)
            
            factors.append({
                "feature": name,
                "factor": desc,
                "impact": impact,
                "contribution_score": round(contrib, 3)
            })
            
        # Sort factors by impact magnitude
        factors.sort(key=lambda x: abs(x["contribution_score"]), reverse=True)
        top_factors = factors[:4] # Top 4 primary factors
        
        # Build manager plain-language explanation string
        pos_list = [f["factor"] for f in top_factors if f["impact"] == "positive"]
        neg_list = [f["factor"] for f in top_factors if f["impact"] == "negative"]
        
        pos_text = "; ".join(pos_list) if pos_list else "Standard baseline score"
        neg_text = "; ".join(neg_list) if neg_list else "No significant negative risk factors"
        
        summary = (
            f"Estimated repayment probability: {prob_pay*100:.1f}%, reachability: {prob_contact*100:.1f}%. "
            f"Recommended: {recommended_action} (Expected recovery: ₹{expected_recovery:,.0f}). "
            f"Key strengths: [{pos_text}]. Risk notes: [{neg_text}]."
        )
        
        return {
            "borrower_id": loan_dict.get("borrower_id"),
            "loan_id": loan_dict.get("id"),
            "repayment_probability": round(prob_pay, 3),
            "contact_probability": round(prob_contact, 3),
            "expected_recovery": round(expected_recovery, 2),
            "recommended_action": recommended_action,
            "factors": top_factors,
            "manager_explanation": summary
        }
