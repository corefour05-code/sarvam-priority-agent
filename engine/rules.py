"""
Regulatory Compliance & Business Rules Gatekeeper.
Enforces RBI debt collection guidelines, calling windows, dispute routing,
hardship alternatives, and promise-to-pay (PTP) scheduling.
"""

from datetime import datetime, date
from typing import Tuple, Dict, Any, Optional
from config import settings

class ComplianceRulesEngine:
    def __init__(self):
        self.window_start = settings.CALLING_WINDOW_START_HOUR
        self.window_end = settings.CALLING_WINDOW_END_HOUR
        
    def evaluate_gatekeeper(self, loan: dict, current_dt: Optional[datetime] = None) -> Tuple[bool, str, str]:
        """
        Evaluates pre-ranking compliance rules.
        Returns: (is_eligible_to_call_now, recommended_action, reason)
        """
        if current_dt is None:
            current_dt = datetime.now()
            
        current_hour = current_dt.hour
        today_date = current_dt.date()
        
        status = str(loan.get("status", "ACTIVE")).upper()
        
        # Rule 1: Already Paid
        if status == "PAID" or float(loan.get("amount_overdue", 0) or 0) <= 0:
            return False, "EXCLUDE_PAID", "Account settled or marked paid. Calling prohibited."
            
        # Rule 2: Claimed Paid / Pending Verification (e.g. 'branch mein kal jama kiya')
        if status == "PENDING_VERIFICATION":
            return False, "HOLD_VERIFY_PAYMENT", "Borrower claimed payment at branch/online. Calling paused pending bank reconciliation."
            
        # Rule 3: Active Dispute
        if status == "DISPUTED":
            return False, "ROUTE_DISPUTE_DESK", f"Account flagged for dispute ({loan.get('dispute_reason', 'Disputed fees')}). Calling suspended; routed to grievance desk."
            
        # Rule 4: Hardship Case
        if status == "HARDSHIP":
            return False, "OFFER_HARDSHIP_RESTRUCTURING", f"Borrower experiencing verified hardship ({loan.get('hardship_reason', 'Medical/Economic emergency')}). Do not pressure; offer restructuring moratorium."
            
        # Rule 5: Promise to Pay (PTP) Scheduling
        p_status = str(loan.get("promise_status", "NONE")).upper()
        ptp_dt = loan.get("active_promise_date")
        
        if ptp_dt is not None:
            if isinstance(ptp_dt, str):
                try:
                    ptp_dt = datetime.fromisoformat(ptp_dt)
                except Exception:
                    ptp_dt = None
            if hasattr(ptp_dt, "to_pydatetime"):
                ptp_dt = ptp_dt.to_pydatetime()
                
        if ptp_dt is not None and p_status == "PENDING":
            ptp_date = ptp_dt.date()
            if ptp_date > today_date:
                days_away = (ptp_date - today_date).days
                return False, "SCHEDULE_FUTURE_PTP", f"Active promise to pay scheduled for {ptp_date} ({days_away} days away, e.g. salary date). Defer calling today."
            elif ptp_date == today_date:
                # Due today! Eligible with high urgency
                return True, "CONTACT_PROMISE_DUE_TODAY", "Commitment date is today. High-priority reminder call."
            else:
                # Lapsed promise without payment
                return True, "CONTACT_BROKEN_PROMISE", f"Promise date ({ptp_date}) has lapsed without confirmed payment. Follow-up required."
                
        # Rule 6: Regulatory Calling Hours Window (08:00 to 19:00)
        if not (self.window_start <= current_hour < self.window_end):
            return False, "DEFER_OUTSIDE_HOURS", f"Current local time ({current_hour:02d}:00) is outside permitted calling hours ({self.window_start:02d}:00 - {self.window_end:02d}:00). Defer outreach."
            
        # Rule 7: Active Eligible Borrower
        return True, "CONTACT_NOW", "Account eligible for standard prioritized recovery contact."
        
    def get_hardship_solution_offer(self, loan: dict) -> Dict[str, Any]:
        """
        Formulates empathetic, regulatory-compliant alternative for borrowers who cannot pay in full.
        """
        overdue = float(loan.get("amount_overdue", 0))
        # Split into 3 manageable installments
        split_amount = round(overdue / 3.0, 2)
        return {
            "hardship_type": "RESTRUCTURED_INSTALLMENTS",
            "message": "We understand your current circumstances. Rather than paying the full amount today, we can split your dues into 3 manageable installments.",
            "split_installments": [
                {"installment": 1, "amount": split_amount, "link_due": "Immediate (via Cashfree)"},
                {"installment": 2, "amount": split_amount, "link_due": "+30 days"},
                {"installment": 3, "amount": split_amount, "link_due": "+60 days"}
            ],
            "moratorium_option": "14-day contact pause with interest freeze while paperwork is submitted."
        }

rules_engine = ComplianceRulesEngine()
