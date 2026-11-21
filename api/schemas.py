"""
Pydantic Schemas for API Requests, Webhooks, and Teammate Integration Contracts.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class CashfreeWebhookPayload(BaseModel):
    order_id: Optional[str] = None
    loan_id: Optional[str] = None
    amount: float = Field(..., gt=0)
    status: str = Field(default="SUCCESS") # SUCCESS, FAILED
    payment_method: str = Field(default="UPI")
    data: Optional[Dict[str, Any]] = None

class SarvamSignalRequest(BaseModel):
    loan_id: str
    raw_text: Optional[str] = None
    detected_intent: Optional[str] = None # CLAIMED_PAID, DISPUTE, HARDSHIP, PTP_SALARY, PARTIAL_PAYMENT
    extracted_amount: Optional[float] = None
    extracted_promise_date: Optional[str] = None
    source: str = Field(default="SARVAM_SPEECH") # SARVAM_SPEECH, SARVAM_VISION_OCR, WHATSAPP

class CreatePaymentLinkRequest(BaseModel):
    loan_id: str
    amount: float = Field(..., gt=0)
    purpose: Optional[str] = "Overdue EMI Recovery"

class QueueResponseItem(BaseModel):
    queue_rank: int
    loan_id: str
    borrower_id: str
    borrower_name: str
    phone: str
    preferred_language: str
    amount_overdue: float
    days_past_due: int
    repayment_probability: float
    contact_probability: float
    expected_recovery: float
    priority_score: float
    recommended_action: str
    manager_explanation: str
    disclosure_statement: str
    is_exploration: bool

class QueueResponse(BaseModel):
    capacity: int
    total_evaluated: int
    eligible_count: int
    deferred_count: int
    summary_metrics: Dict[str, Any]
    queue: List[QueueResponseItem]
