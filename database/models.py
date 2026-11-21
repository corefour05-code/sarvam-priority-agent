"""
SQLAlchemy ORM models for Debt Recovery System.
Encapsulates Borrowers, Loans, Promises, Payments, and Real-Time Event Signals.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from database.connection import Base

class Borrower(Base):
    __tablename__ = "borrowers"

    id = Column(String(36), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    phone = Column(String(20), nullable=False, index=True)
    preferred_language = Column(String(10), default="hi") # Hindi, Tamil, Telugu, English
    
    # Protected attributes - STRICTLY AUDIT ONLY, NEVER IN ML FEATURES
    gender = Column(String(20), default="unspecified")
    caste_category = Column(String(50), default="general")
    religion = Column(String(50), default="unspecified")
    pincode = Column(String(10), default="560001")
    created_at = Column(DateTime, default=datetime.utcnow)

    loans = relationship("Loan", back_populates="borrower")


class Loan(Base):
    __tablename__ = "loans"

    id = Column(String(36), primary_key=True, index=True) # Loan number
    borrower_id = Column(String(36), ForeignKey("borrowers.id"), nullable=False)
    
    # Financial metrics
    principal_amount = Column(Float, nullable=False)
    amount_overdue = Column(Float, nullable=False, index=True)
    days_past_due = Column(Integer, nullable=False, default=0, index=True) # DPD
    installments_total = Column(Integer, default=12)
    installments_paid = Column(Integer, default=0)
    
    # Payment History
    last_payment_date = Column(DateTime, nullable=True)
    last_payment_amount = Column(Float, default=0.0)
    
    # Operational Status
    # ACTIVE, PAID, PENDING_VERIFICATION, DISPUTED, HARDSHIP, RESTRUCTURED
    status = Column(String(30), default="ACTIVE", index=True)
    dispute_reason = Column(Text, nullable=True)
    hardship_reason = Column(Text, nullable=True)
    
    # Commitment / Promise to Pay (PTP)
    # NONE, PENDING, FULFILLED, BROKEN
    active_promise_date = Column(DateTime, nullable=True)
    active_promise_amount = Column(Float, default=0.0)
    promise_status = Column(String(20), default="NONE")
    promises_made_count = Column(Integer, default=0)
    promises_fulfilled_count = Column(Integer, default=0)
    
    # Contact & Reachability history
    contact_attempts_count = Column(Integer, default=0)
    contact_responses_count = Column(Integer, default=0)
    last_contact_outcome = Column(String(50), default="NEVER_CONTACTED")
    last_contact_timestamp = Column(DateTime, nullable=True)
    
    # Model Scores (computed dynamically)
    payment_probability = Column(Float, default=0.0)
    contact_probability = Column(Float, default=0.5)
    expected_recovery = Column(Float, default=0.0)
    priority_score = Column(Float, default=0.0)
    recommended_action = Column(String(50), default="CONTACT_NOW")
    explanation_summary = Column(Text, nullable=True)
    
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    borrower = relationship("Borrower", back_populates="loans")
    payments = relationship("Payment", back_populates="loan")
    signals = relationship("SignalEvent", back_populates="loan")
    call_logs = relationship("CallLog", back_populates="loan")


class Payment(Base):
    __tablename__ = "payments"

    id = Column(String(36), primary_key=True, index=True)
    loan_id = Column(String(36), ForeignKey("loans.id"), nullable=False)
    order_id = Column(String(100), nullable=True) # Cashfree order ID
    payment_link_id = Column(String(100), nullable=True)
    amount = Column(Float, nullable=False)
    # SUCCESS, PENDING, FAILED
    status = Column(String(20), default="PENDING")
    payment_method = Column(String(50), default="UPI")
    verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    loan = relationship("Loan", back_populates="payments")


class SignalEvent(Base):
    __tablename__ = "signal_events"

    id = Column(String(36), primary_key=True, index=True)
    loan_id = Column(String(36), ForeignKey("loans.id"), nullable=False)
    # SARVAM_SPEECH, SARVAM_VISION_OCR, CASHFREE_WEBHOOK, WHATSAPP, MANUAL
    source = Column(String(50), nullable=False)
    raw_text = Column(Text, nullable=True)
    # PTP_SALARY, CLAIMED_PAID, DISPUTE, HARDSHIP, PARTIAL_PAYMENT
    extracted_intent = Column(String(50), nullable=True)
    extracted_amount = Column(Float, nullable=True)
    extracted_date = Column(DateTime, nullable=True)
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    loan = relationship("Loan", back_populates="signals")


class CallLog(Base):
    __tablename__ = "call_logs"

    id = Column(String(36), primary_key=True, index=True)
    loan_id = Column(String(36), ForeignKey("loans.id"), nullable=False)
    caller_name = Column(String(50), default="Agent_Bulbul")
    timestamp = Column(DateTime, default=datetime.utcnow)
    duration_seconds = Column(Integer, default=0)
    answered = Column(Boolean, default=False)
    outcome = Column(String(50), default="UNANSWERED")
    disclosure_read = Column(Boolean, default=True) # Mandatory regulatory compliance
    hardship_offered = Column(Boolean, default=False)

    loan = relationship("Loan", back_populates="call_logs")
