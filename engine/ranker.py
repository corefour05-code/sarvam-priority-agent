"""
Economic Prioritization & Exploration-Exploitation Queue Ranker.
Ranks eligible borrowers using expected recovery per call cost,
while allocating an exploration budget to prevent sampling bias.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from config import settings
from ml.model import predictor
from ml.explainer import ScoreExplainer
from ml.validator import DataIntegrityAuditor
from engine.rules import rules_engine

class PriorityRanker:
    def __init__(self, pred_engine=None):
        self.predictor = pred_engine or predictor
        self.explainer = ScoreExplainer(self.predictor)
        self.call_cost = settings.CALL_COST_INR
        self.exploration_ratio = settings.EXPLORATION_RATIO
        
    def rank_portfolio(
        self,
        loans: Any,
        capacity: int = 60,
        current_dt: Optional[datetime] = None,
        enable_exploration: bool = True
    ) -> Dict[str, Any]:
        """
        Ranks an entire portfolio of overdue accounts.
        Accepts raw lists of dicts, pandas DataFrames, or file paths.
        Validates, cleans, flags missing values, and quarantines defective rows to prevent false results.
        """
        if current_dt is None:
            current_dt = datetime.now()
            
        # Step 0: Ingestion, Column Normalization & Integrity Auditing
        clean_res = DataIntegrityAuditor.validate_and_clean_portfolio(loans)
        valid_loans = clean_res["valid_records"]
        quarantined_records = clean_res["quarantined_records"]
        data_health = clean_res["audit_report"]
        
        eligible_candidates = []
        deferred_accounts = []
        
        # Format quarantined accounts with clear manager-visible reasons
        formatted_quarantined = []
        for q in quarantined_records:
            issues = "; ".join([f["issue"] for f in q.get("data_validation_flags", [])])
            formatted_quarantined.append({
                "loan_id": q.get("id"),
                "borrower_name": q.get("name"),
                "phone": q.get("phone"),
                "amount_overdue": q.get("amount_overdue"),
                "status": "DATA_QUARANTINE",
                "recommended_action": "QUARANTINE_DATA_DEFECT",
                "reason": f"Flagged for data integrity issue: [{issues}]. Excluded to prevent false dialing.",
                "priority_score": 0.0,
                "expected_recovery": 0.0
            })
            
        # Step 1: Compliance & Business Gatekeeper Filtering on Valid Records
        for loan in valid_loans:
            is_eligible, action, reason = rules_engine.evaluate_gatekeeper(loan, current_dt)
            
            if not is_eligible:
                deferred_accounts.append({
                    "loan_id": loan.get("id"),
                    "borrower_id": loan.get("borrower_id"),
                    "amount_overdue": loan.get("amount_overdue"),
                    "status": loan.get("status"),
                    "recommended_action": action,
                    "reason": reason,
                    "priority_score": 0.0,
                    "expected_recovery": 0.0
                })
                continue
                
            eligible_candidates.append((loan, action))
            
        if not eligible_candidates:
            return {
                "capacity": capacity,
                "total_accounts_evaluated": data_health["total_records_ingested"],
                "valid_records_count": data_health["valid_records_count"],
                "quarantined_count": data_health["quarantined_records_count"],
                "eligible_for_contact": 0,
                "deferred_count": len(deferred_accounts),
                "data_health_report": data_health,
                "quarantined_accounts": formatted_quarantined,
                "ranked_queue": [],
                "deferred_accounts": deferred_accounts[:50],
                "summary_metrics": {
                    "total_expected_recovery_inr": 0.0,
                    "average_repayment_probability": 0.0,
                    "average_contact_probability": 0.0,
                    "estimated_calls_answered": 0,
                    "estimated_payers_reached": 0
                }
            }
            
        # Step 2: Vectorized Batch Probability Estimation (Ultra-fast < 0.05s)
        eligible_loans = [item[0] for item in eligible_candidates]
        eligible_actions = [item[1] for item in eligible_candidates]
        
        prob_pay_array = self.predictor.predict_batch_repayment_probabilities(eligible_loans)
        
        scored_candidates = []
        for i, loan in enumerate(eligible_loans):
            prob_pay = float(prob_pay_array[i])
            prob_contact = self.predictor.predict_contact_probability(loan)
            amount = float(loan.get("amount_overdue", 0))
            expected_recovery = prob_pay * amount
            
            # Step 3: Economic Priority Formula
            # Priority = (P(pay) * Amount * P(contact)) / Cost
            base_score = (expected_recovery * prob_contact) / max(0.1, self.call_cost)
            
            action = eligible_actions[i]
            if action == "CONTACT_PROMISE_DUE_TODAY":
                base_score *= 1.50
            elif action == "CONTACT_BROKEN_PROMISE":
                base_score *= 1.25
                
            candidate = {
                "loan_dict": loan,
                "loan_id": loan.get("id"),
                "borrower_id": loan.get("borrower_id"),
                "borrower_name": loan.get("name", "Borrower"),
                "phone": loan.get("phone", "N/A"),
                "preferred_language": loan.get("preferred_language", "hi"),
                "amount_overdue": amount,
                "days_past_due": loan.get("days_past_due"),
                "repayment_probability": round(prob_pay, 4),
                "contact_probability": round(prob_contact, 4),
                "expected_recovery": round(expected_recovery, 2),
                "priority_score": round(base_score, 2),
                "recommended_action": action,
                "prerequisites_met": loan.get("prerequisites_met", {"amount_met": True, "date_met": True, "name_met": True, "contact_met": True}),
                "data_validation_flags": loan.get("data_validation_flags", []),
                "is_exploration": False,
                "disclosure_statement": settings.MANDATORY_CALL_RECORDING_DISCLOSURE
            }
            scored_candidates.append(candidate)
            
        # Step 4: Ranking with Exploration-Exploitation
        scored_candidates.sort(key=lambda x: x["priority_score"], reverse=True)
        
        if not enable_exploration or len(scored_candidates) <= capacity:
            final_queue = scored_candidates[:capacity]
        else:
            num_explore = max(1, int(capacity * self.exploration_ratio))
            num_exploit = capacity - num_explore
            
            top_exploit = scored_candidates[:num_exploit]
            remaining = scored_candidates[num_exploit:]
            
            uncalled_pool = [c for c in remaining if c.get("days_past_due", 0) <= 60]
            if len(uncalled_pool) >= num_explore:
                np.random.seed(42)
                explore_indices = np.random.choice(len(uncalled_pool), size=num_explore, replace=False)
                explore_picks = [uncalled_pool[i] for i in explore_indices]
            else:
                explore_picks = remaining[:num_explore]
                
            for ep in explore_picks:
                ep["is_exploration"] = True
                
            final_queue = top_exploit + explore_picks
            final_queue.sort(key=lambda x: x["priority_score"], reverse=True)
            
        # Step 5: Compute Detailed Manager Explanations ONLY for final top queue
        for idx, item in enumerate(final_queue, 1):
            item["queue_rank"] = idx
            explanation = self.explainer.explain(
                loan_dict=item["loan_dict"],
                prob_pay=item["repayment_probability"],
                prob_contact=item["contact_probability"],
                expected_recovery=item["expected_recovery"],
                recommended_action=item["recommended_action"]
            )
            item["factors"] = explanation["factors"]
            item["manager_explanation"] = explanation["manager_explanation"]
            if item.get("is_exploration"):
                item["manager_explanation"] += " [Exploration allocation to refresh portfolio intelligence]"
            del item["loan_dict"] # Clean internal reference
            
        total_expected_recovery = sum(item["expected_recovery"] for item in final_queue)
        avg_pay_prob = (sum(item["repayment_probability"] for item in final_queue) / max(1, len(final_queue)))
        avg_contact_prob = (sum(item["contact_probability"] for item in final_queue) / max(1, len(final_queue)))
        
        return {
            "capacity": capacity,
            "total_accounts_evaluated": data_health["total_records_ingested"],
            "valid_records_count": data_health["valid_records_count"],
            "quarantined_count": data_health["quarantined_records_count"],
            "eligible_for_contact": len(eligible_candidates),
            "deferred_count": len(deferred_accounts),
            "data_health_report": data_health,
            "quarantined_accounts": formatted_quarantined,
            "ranked_queue": final_queue,
            "deferred_accounts": deferred_accounts[:50], # sample of deferred
            "summary_metrics": {
                "total_expected_recovery_inr": round(total_expected_recovery, 2),
                "average_repayment_probability": round(avg_pay_prob, 3),
                "average_contact_probability": round(avg_contact_prob, 3),
                "estimated_calls_answered": round(len(final_queue) * avg_contact_prob, 1),
                "estimated_payers_reached": round(len(final_queue) * avg_pay_prob * avg_contact_prob, 1)
            }
        }

ranker = PriorityRanker()
