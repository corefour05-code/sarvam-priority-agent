"""
Continuous DB Monitor & Autonomous State Machine Engine.
Monitors loan records, receives Cashfree payment webhooks & Sarvam speech intents,
updates commitment statuses, and re-ranks the queue dynamically in real time.
Directly powers: 'The queue re-forms in front of us. A payment lands, a promise lapses...'
"""

from datetime import datetime, date
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session, joinedload
from database.connection import SessionLocal
from database.models import Loan, Borrower, Payment, SignalEvent, CallLog
from engine.ranker import ranker
from integrations.cashfree_client import cashfree_client
import uuid

class PortfolioMonitor:
    def __init__(self):
        self.last_queue_cache = []
        self.change_history = []
        
    def get_current_portfolio_dicts(self, db: Session) -> List[Dict[str, Any]]:
        """
        Retrieves all active overdue loans joined with borrower information in a single query.
        """
        loans = db.query(Loan).options(joinedload(Loan.borrower)).all()
        portfolio = []
        for l in loans:
            b = l.borrower
            portfolio.append({
                "id": l.id,
                "borrower_id": l.borrower_id,
                "name": b.name if b else "Borrower",
                "phone": b.phone if b else "N/A",
                "preferred_language": b.preferred_language if b else "hi",
                "gender": b.gender if b else "unspecified",
                "religion": b.religion if b else "unspecified",
                "caste_category": b.caste_category if b else "general",
                "pincode": b.pincode if b else "560001",
                "principal_amount": l.principal_amount,
                "amount_overdue": l.amount_overdue,
                "days_past_due": l.days_past_due,
                "installments_total": l.installments_total,
                "installments_paid": l.installments_paid,
                "last_payment_date": l.last_payment_date,
                "last_payment_amount": l.last_payment_amount,
                "status": l.status,
                "dispute_reason": l.dispute_reason,
                "hardship_reason": l.hardship_reason,
                "active_promise_date": l.active_promise_date,
                "active_promise_amount": l.active_promise_amount,
                "promise_status": l.promise_status,
                "promises_made_count": l.promises_made_count,
                "promises_fulfilled_count": l.promises_fulfilled_count,
                "contact_attempts_count": l.contact_attempts_count,
                "contact_responses_count": l.contact_responses_count,
                "last_contact_outcome": l.last_contact_outcome
            })
        return portfolio
        
    def re_rank_and_persist(self, capacity: int = 60, current_dt: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Evaluates full portfolio, recalculates probabilities, updates DB and persists top queue.
        """
        if current_dt is None:
            current_dt = datetime.now()
            
        db = SessionLocal()
        try:
            portfolio = self.get_current_portfolio_dicts(db)
            result = ranker.rank_portfolio(portfolio, capacity=capacity, current_dt=current_dt)
            ranked_queue = result["ranked_queue"]
            
            # Map of updated loans
            queue_map = {item["loan_id"]: item for item in ranked_queue}
            
            # Persist computed scores only for top ranked loans (ultra-fast real-time response)
            top_loan_ids = list(queue_map.keys())
            ranked_loans = db.query(Loan).filter(Loan.id.in_(top_loan_ids)).all()
            for l in ranked_loans:
                info = queue_map[l.id]
                l.priority_score = info["priority_score"]
                l.payment_probability = info["repayment_probability"]
                l.contact_probability = info["contact_probability"]
                l.expected_recovery = info["expected_recovery"]
                l.recommended_action = info["recommended_action"]
                l.explanation_summary = info["manager_explanation"]
            db.commit()
            
            self.last_queue_cache = ranked_queue
            return result
        finally:
            db.close()
            
    def handle_payment_event(self, loan_id: str, amount: float, order_id: str = None, payment_method: str = "UPI") -> Dict[str, Any]:
        """
        Triggered when a Cashfree payment webhook lands.
        Updates balance, marks promise fulfilled, re-ranks queue immediately.
        """
        db = SessionLocal()
        try:
            loan = db.query(Loan).filter(Loan.id == loan_id).first()
            if not loan:
                return {"success": False, "error": f"Loan {loan_id} not found"}
                
            old_balance = loan.amount_overdue
            new_balance = max(0.0, old_balance - amount)
            is_full_settlement = (new_balance <= 0.0)
            
            # Record payment in ledger
            pmt = Payment(
                id=f"pmt_{uuid.uuid4().hex[:8]}",
                loan_id=loan.id,
                order_id=order_id or f"ORD_{uuid.uuid4().hex[:6]}",
                amount=amount,
                status="SUCCESS",
                payment_method=payment_method,
                verified=True,
                created_at=datetime.utcnow()
            )
            db.add(pmt)
            
            # Update loan state
            loan.amount_overdue = new_balance
            loan.last_payment_date = datetime.utcnow()
            loan.last_payment_amount = amount
            loan.installments_paid += 1
            
            # Commitment tracking: Reward fulfilled promise
            if loan.promise_status in ["PENDING", "BROKEN"]:
                loan.promise_status = "FULFILLED"
                loan.promises_fulfilled_count += 1
                
            if is_full_settlement:
                loan.status = "PAID"
                loan.priority_score = 0.0
                loan.recommended_action = "EXCLUDE_PAID"
                action_summary = f"Full settlement of ₹{amount:,.0f} verified via Cashfree. Account settled and removed from queue."
            else:
                action_summary = f"Partial payment of ₹{amount:,.0f} verified. Outstanding balance reduced to ₹{new_balance:,.0f}. Promise credit added."
                
            db.commit()
            
            # Log event change
            event_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "CASHFREE_PAYMENT_LANDED",
                "loan_id": loan.id,
                "borrower_id": loan.borrower_id,
                "amount": amount,
                "new_balance": new_balance,
                "action_taken": action_summary
            }
            self.change_history.append(event_log)
            
            # Re-rank queue immediately in real time
            new_ranking = self.re_rank_and_persist()
            
            return {
                "success": True,
                "event": event_log,
                "action_summary": action_summary,
                "is_full_settlement": is_full_settlement,
                "top_5_queue": new_ranking["ranked_queue"][:5]
            }
        finally:
            db.close()
            
    def handle_intent_signal(self, loan_id: str, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Triggered when a Sarvam voice call or OCR note extracts new conversational intent.
        """
        db = SessionLocal()
        try:
            loan = db.query(Loan).filter(Loan.id == loan_id).first()
            if not loan:
                return {"success": False, "error": f"Loan {loan_id} not found"}
                
            detected_intent = intent_data.get("detected_intent")
            raw_text = intent_data.get("raw_text", "")
            action_taken = ""
            payment_link_data = None
            
            # Save raw signal event
            signal = SignalEvent(
                id=f"sig_{uuid.uuid4().hex[:8]}",
                loan_id=loan.id,
                source=intent_data.get("source", "SARVAM_SPEECH"),
                raw_text=raw_text,
                extracted_intent=detected_intent,
                extracted_amount=intent_data.get("extracted_amount"),
                extracted_date=datetime.fromisoformat(intent_data["extracted_promise_date"]) if intent_data.get("extracted_promise_date") else None,
                processed=True
            )
            db.add(signal)
            
            if detected_intent == "CLAIMED_PAID":
                loan.status = "PENDING_VERIFICATION"
                action_taken = "Borrower claimed payment was made at branch. Calling held pending manual verification."
                
            elif detected_intent == "DISPUTE":
                loan.status = "DISPUTED"
                loan.dispute_reason = raw_text or "Disputed charges / interest"
                action_taken = "Borrower disputed charges. Account frozen from calling queue and escalated to grievance desk."
                
            elif detected_intent == "HARDSHIP":
                loan.status = "HARDSHIP"
                loan.hardship_reason = raw_text or "Emergency hardship declared"
                action_taken = "Hardship detected. Switched to empathy mode: offering 3-part split restructuring."
                
            elif detected_intent == "PTP_SALARY":
                ptp_iso = intent_data.get("extracted_promise_date")
                if ptp_iso:
                    ptp_dt = datetime.fromisoformat(ptp_iso)
                    loan.active_promise_date = ptp_dt
                    loan.promise_status = "PENDING"
                    loan.promises_made_count += 1
                    action_taken = f"Salary date identified ({ptp_dt.strftime('%d %b %Y')}). Calling deferred; follow-up scheduled for commitment date."
                    
            elif detected_intent == "PARTIAL_PAYMENT":
                amt = float(intent_data.get("extracted_amount") or loan.amount_overdue)
                b = loan.borrower
                payment_link_data = cashfree_client.create_payment_link(
                    loan_id=loan.id,
                    borrower_name=b.name if b else "Borrower",
                    phone=b.phone if b else "9876543210",
                    amount=amt
                )
                action_taken = f"Borrower agreed to pay ₹{amt:,.0f} now. Instant Cashfree payment link generated and dispatched."
                
            db.commit()
            
            event_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "SARVAM_INTENT_SIGNAL",
                "loan_id": loan.id,
                "intent": detected_intent,
                "action_taken": action_taken,
                "payment_link": payment_link_data.get("payment_url") if payment_link_data else None
            }
            self.change_history.append(event_log)
            
            # Re-rank queue immediately
            new_ranking = self.re_rank_and_persist()
            
            return {
                "success": True,
                "event": event_log,
                "action_taken": action_taken,
                "payment_link": payment_link_data,
                "top_5_queue": new_ranking["ranked_queue"][:5]
            }
        finally:
            db.close()
            
    def check_lapsed_promises(self, current_dt: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Autonomous cron check: checks for promises whose date has passed without payment.
        """
        if current_dt is None:
            current_dt = datetime.now()
            
        db = SessionLocal()
        try:
            today_date = current_dt.date()
            pending_promises = db.query(Loan).filter(
                Loan.promise_status == "PENDING",
                Loan.active_promise_date != None
            ).all()
            
            lapsed_loans = []
            for loan in pending_promises:
                if loan.active_promise_date.date() < today_date:
                    loan.promise_status = "BROKEN"
                    lapsed_loans.append(loan.id)
                    
            if lapsed_loans:
                db.commit()
                event_log = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "PROMISES_LAPSED",
                    "count": len(lapsed_loans),
                    "loan_ids": lapsed_loans[:10],
                    "action_taken": f"{len(lapsed_loans)} promises lapsed. Broken-promise penalty applied; prioritized for escalation call."
                }
                self.change_history.append(event_log)
                self.re_rank_and_persist(current_dt=current_dt)
                return {"lapsed_count": len(lapsed_loans), "loans": lapsed_loans}
            return {"lapsed_count": 0, "loans": []}
        finally:
            db.close()

monitor = PortfolioMonitor()
