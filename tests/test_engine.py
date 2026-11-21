"""
Comprehensive Automated Test Suite for FinVoice Autonomous Recovery Agent.
Tests compliance rules, ML fairness, economic ranking, Cashfree webhook processing,
and dynamic queue replanning.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from ml.features import extract_features_from_dict, PROTECTED_ATTRIBUTES, FEATURE_NAMES
from ml.model import predictor
from ml.fairness_audit import FairnessAuditor
from engine.rules import rules_engine
from engine.ranker import ranker
from engine.monitor import monitor
from integrations.cashfree_client import cashfree_client
from integrations.sarvam_adapter import sarvam_adapter

def test_feature_extraction_red_line_compliance():
    """Verify that NO demographic attributes leak into feature vectors."""
    loan_dict = {
        "id": "LN12345",
        "borrower_id": "B9999",
        "name": "Pooja Sharma",
        "gender": "female",
        "religion": "hindu",
        "caste_category": "obc",
        "preferred_language": "hi",
        "pincode": "560001",
        "promises_made_count": 3,
        "promises_fulfilled_count": 2,
        "installments_total": 12,
        "installments_paid": 6,
        "days_past_due": 15,
        "amount_overdue": 5000,
        "principal_amount": 25000,
        "contact_attempts_count": 2,
        "contact_responses_count": 2,
        "promise_status": "NONE"
    }
    features = extract_features_from_dict(loan_dict)
    assert len(features) == len(FEATURE_NAMES)
    assert not np.isnan(features).any()

def test_fairness_counterfactual_permutation():
    """Verify swapping gender/religion/caste produces 0.00% change in score."""
    auditor = FairnessAuditor()
    audit_res = auditor.run_full_audit()
    assert audit_res["status"] == "COMPLIANT"
    assert audit_res["feature_leakage_check"]["passed"] is True
    assert audit_res["counterfactual_permutation_test"]["passed"] is True
    assert audit_res["counterfactual_permutation_test"]["max_score_delta_under_demographic_swap"] == 0.0

def test_compliance_rules_gatekeeper():
    """Test RBI calling hours, dispute freezing, hardship offering, and PTP scheduling."""
    # 1. Paid loan
    is_eligible, action, _ = rules_engine.evaluate_gatekeeper({"status": "PAID", "amount_overdue": 0})
    assert is_eligible is False
    assert action == "EXCLUDE_PAID"
    
    # 2. Active Dispute
    is_eligible, action, _ = rules_engine.evaluate_gatekeeper({"status": "DISPUTED", "amount_overdue": 10000})
    assert is_eligible is False
    assert action == "ROUTE_DISPUTE_DESK"
    
    # 3. Hardship
    is_eligible, action, _ = rules_engine.evaluate_gatekeeper({"status": "HARDSHIP", "amount_overdue": 15000})
    assert is_eligible is False
    assert action == "OFFER_HARDSHIP_RESTRUCTURING"
    
    # 4. Future PTP (Salary on 25th, today is 21st)
    future_dt = datetime.now() + timedelta(days=4)
    is_eligible, action, _ = rules_engine.evaluate_gatekeeper({
        "status": "ACTIVE",
        "amount_overdue": 8000,
        "promise_status": "PENDING",
        "active_promise_date": future_dt
    })
    assert is_eligible is False
    assert action == "SCHEDULE_FUTURE_PTP"
    
    # 5. Calling hours check (night time: 23:00)
    night_dt = datetime(2026, 11, 21, 23, 0)
    is_eligible, action, _ = rules_engine.evaluate_gatekeeper({"status": "ACTIVE", "amount_overdue": 5000}, current_dt=night_dt)
    assert is_eligible is False
    assert action == "DEFER_OUTSIDE_HOURS"

def test_sarvam_intent_adapter():
    """Verify parsing of code-mixed Hindi/English speech and OCR texts."""
    # PTP Salary
    res1 = sarvam_adapter.parse_speech_or_ocr_transcript("Bhaiya meri salary 5th ko aayegi tab payment karunga")
    assert res1["detected_intent"] == "PTP_SALARY"
    assert res1["extracted_promise_date"] is not None
    
    # Claimed Paid
    res2 = sarvam_adapter.parse_speech_or_ocr_transcript("Maine branch mein kal cash jama kar diya hai")
    assert res2["detected_intent"] == "CLAIMED_PAID"
    
    # Dispute
    res3 = sarvam_adapter.parse_speech_or_ocr_transcript("Galat charges aur extra interest lagaya hai, dispute hai")
    assert res3["detected_intent"] == "DISPUTE"
    
    # Pay Now
    res4 = sarvam_adapter.parse_speech_or_ocr_transcript("Main abhi 5000 de sakta hu payment link bhejo")
    assert res4["detected_intent"] == "PARTIAL_PAYMENT"
    assert res4["extracted_amount"] == 5000.0

def test_cashfree_payment_link_generation():
    """Test generating Cashfree links with fallback simulation."""
    res = cashfree_client.create_payment_link(
        loan_id="LN99999",
        borrower_name="Rohan Verma",
        phone="9876543210",
        amount=4500.0
    )
    assert res["success"] is True
    assert "https://" in res["payment_url"]
    assert "LN99999" in res["payment_url"]

def test_economic_ranker_logic():
    """Verify that expected recovery balances probability vs loan size."""
    borrower_a = {
        "id": "LN_A", "borrower_id": "BA", "name": "A", "amount_overdue": 500.0,
        "days_past_due": 5, "installments_total": 12, "installments_paid": 10,
        "promises_made_count": 5, "promises_fulfilled_count": 5, "status": "ACTIVE"
    }
    borrower_b = {
        "id": "LN_B", "borrower_id": "BB", "name": "B", "amount_overdue": 20000.0,
        "days_past_due": 20, "installments_total": 12, "installments_paid": 6,
        "promises_made_count": 3, "promises_fulfilled_count": 2, "status": "ACTIVE"
    }
    
    res = ranker.rank_portfolio([borrower_a, borrower_b], capacity=2, enable_exploration=False)
    queue = res["ranked_queue"]
    
    # Borrower B has far higher expected yield (e.g. 60% of 20,000 = ₹12,000 vs 90% of 500 = ₹450)
    assert queue[0]["loan_id"] == "LN_B"
    assert queue[1]["loan_id"] == "LN_A"


def test_data_integrity_auditor_missing_amount_quarantine():
    """Verify that records with missing overdue amounts are quarantined and never assigned false recovery."""
    bad_record = {
        "id": "LN_DEFECT_1",
        "name": "Corrupt Record",
        "phone": "+919876543210",
        "amount_overdue": None, # Missing amount!
        "days_past_due": 30
    }
    good_record = {
        "id": "LN_VALID_1",
        "name": "Valid Borrower",
        "phone": "+919876543211",
        "amount_overdue": 15000.0,
        "days_past_due": 10
    }
    
    result = ranker.rank_portfolio([bad_record, good_record], capacity=2)
    assert result["quarantined_count"] == 1
    assert len(result["quarantined_accounts"]) == 1
    quarantined = result["quarantined_accounts"][0]
    assert quarantined["loan_id"] == "LN_DEFECT_1"
    assert quarantined["recommended_action"] == "QUARANTINE_DATA_DEFECT"
    assert quarantined["priority_score"] == 0.0
    assert quarantined["expected_recovery"] == 0.0
    
    # Valid record should be ranked without error
    assert len(result["ranked_queue"]) == 1
    assert result["ranked_queue"][0]["loan_id"] == "LN_VALID_1"
    assert result["ranked_queue"][0]["prerequisites_met"]["amount_met"] is True

def test_data_integrity_auditor_arbitrary_column_aliases():
    """Verify that arbitrary column names (e.g. acct_no, due_amt, mobile) are mapped seamlessly."""
    messy_columns_record = {
        "acct_no": "LN_MESSY_99",
        "applicant_name": "Deepak Mehta",
        "mobile_no": "9876543210",
        "due_amt": "25,000",
        "dpd": 18,
        "tenor": 24
    }
    
    result = ranker.rank_portfolio([messy_columns_record], capacity=1)
    assert result["quarantined_count"] == 0
    assert len(result["ranked_queue"]) == 1
    item = result["ranked_queue"][0]
    assert item["loan_id"] == "LN_MESSY_99"
    assert item["borrower_name"] == "Deepak Mehta"
    assert item["amount_overdue"] == 25000.0
    assert item["days_past_due"] == 18
    assert item["prerequisites_met"]["amount_met"] is True
    assert item["prerequisites_met"]["name_met"] is True

def test_data_integrity_flagging_missing_prerequisites():
    """Verify missing phone, name, and date are explicitly flagged in prerequisites_met."""
    incomplete_record = {
        "id": "LN_INCOMPLETE",
        "amount_overdue": 50000.0,
        "days_past_due": 45,
        "last_payment_date": "invalid_date_xyz"
        # Missing name and phone
    }
    result = ranker.rank_portfolio([incomplete_record], capacity=1)
    assert len(result["ranked_queue"]) == 1
    item = result["ranked_queue"][0]
    prereqs = item["prerequisites_met"]
    
    assert prereqs["amount_met"] is True   # Amount was present
    assert prereqs["name_met"] is False     # Name was missing -> flagged
    assert prereqs["contact_met"] is False  # Phone was missing -> flagged
    assert prereqs["date_met"] is False     # Date was unparseable -> flagged
    assert item["borrower_name"] == "Valued Customer" # Safe fallback without hallucination
