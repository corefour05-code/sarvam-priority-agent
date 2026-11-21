"""
Fair & Compliant Feature Engineering for Debt Recovery.
Extracts behavioral, commitment, and recency signals.
STRICT RED LINE: Demographic attributes (gender, religion, caste, name, pincode, language)
are explicitly banned from feature vectors.
"""

from datetime import datetime
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "promise_fulfillment_ratio",
    "has_prior_promises",
    "installment_repay_ratio",
    "dpd_normalized",
    "overdue_to_principal_ratio",
    "recency_days_normalized",
    "contact_response_rate",
    "broken_promise_penalty",
    "ptp_due_today_signal"
]

PROTECTED_ATTRIBUTES = [
    "gender", "religion", "caste_category", "caste", "name", "pincode", "preferred_language", "language", "phone"
]

def extract_features_from_dict(row: dict, ref_date: datetime = None) -> np.ndarray:
    """
    Extracts a 1D feature array from a single loan/borrower dictionary.
    Guarantees no protected attributes are included.
    """
    if ref_date is None:
        ref_date = datetime.now()
        
    promises_made = float(row.get("promises_made_count", 0) or 0)
    promises_fulfilled = float(row.get("promises_fulfilled_count", 0) or 0)
    promise_fulfillment_ratio = (promises_fulfilled / promises_made) if promises_made > 0 else 0.5
    has_prior_promises = 1.0 if promises_made > 0 else 0.0
    
    inst_total = float(row.get("installments_total", 12) or 12)
    inst_paid = float(row.get("installments_paid", 0) or 0)
    installment_repay_ratio = inst_paid / max(1.0, inst_total)
    
    dpd = float(row.get("days_past_due", 0) or 0)
    dpd_normalized = min(1.0, dpd / 120.0)
    
    principal = float(row.get("principal_amount", 10000) or 10000)
    overdue = float(row.get("amount_overdue", 0) or 0)
    overdue_to_principal_ratio = min(1.0, overdue / max(1.0, principal))
    
    last_pay_dt = row.get("last_payment_date")
    if last_pay_dt is not None:
        if isinstance(last_pay_dt, str):
            try:
                last_pay_dt = datetime.fromisoformat(last_pay_dt)
            except Exception:
                last_pay_dt = None
        if hasattr(last_pay_dt, "to_pydatetime"):
            last_pay_dt = last_pay_dt.to_pydatetime()
            
    if last_pay_dt is not None and not pd.isna(last_pay_dt):
        days_since_pay = max(0, (ref_date - last_pay_dt).days)
    else:
        days_since_pay = 120
    recency_days_normalized = min(1.0, days_since_pay / 120.0)
    
    attempts = float(row.get("contact_attempts_count", 0) or 0)
    responses = float(row.get("contact_responses_count", 0) or 0)
    contact_response_rate = (responses / attempts) if attempts > 0 else 0.4
    
    p_status = str(row.get("promise_status", "NONE"))
    broken_promise_penalty = 1.0 if p_status == "BROKEN" else 0.0
    
    ptp_dt = row.get("active_promise_date")
    if ptp_dt is not None:
        if isinstance(ptp_dt, str):
            try:
                ptp_dt = datetime.fromisoformat(ptp_dt)
            except Exception:
                ptp_dt = None
        if hasattr(ptp_dt, "to_pydatetime"):
            ptp_dt = ptp_dt.to_pydatetime()
            
    ptp_due_today = 0.0
    if ptp_dt is not None and not pd.isna(ptp_dt) and p_status == "PENDING":
        if ptp_dt.date() == ref_date.date():
            ptp_due_today = 1.0
            
    features = np.array([
        promise_fulfillment_ratio,
        has_prior_promises,
        installment_repay_ratio,
        dpd_normalized,
        overdue_to_principal_ratio,
        recency_days_normalized,
        contact_response_rate,
        broken_promise_penalty,
        ptp_due_today
    ], dtype=np.float32)
    
    return features

def extract_features_dataframe(df: pd.DataFrame, ref_date: datetime = None) -> pd.DataFrame:
    """
    Extracts numerical feature matrix from a pandas dataframe.
    """
    feature_rows = []
    for _, row in df.iterrows():
        f = extract_features_from_dict(row.to_dict(), ref_date)
        feature_rows.append(f)
        
    df_features = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    return df_features
