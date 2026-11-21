"""
FastAPI Backend Application for FinVoice Autonomous Recovery Agent.
Provides real-time endpoints for Cashfree webhooks, Sarvam speech/OCR signal ingestion,
dynamic queue re-ranking, and regulatory fairness auditing.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from database.connection import init_db
from engine.monitor import monitor
from ml.fairness_audit import FairnessAuditor
from integrations.cashfree_client import cashfree_client
from integrations.sarvam_adapter import sarvam_adapter
from api.schemas import (
    CashfreeWebhookPayload,
    SarvamSignalRequest,
    CreatePaymentLinkRequest,
    QueueResponse
)

app = FastAPI(
    title=settings.APP_NAME,
    description="Autonomous Event-Driven Debt Recovery Prioritization & Replanning Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

auditor = FairnessAuditor()

@app.on_event("startup")
async def startup_event():
    init_db()
    # Initial ranking on startup
    monitor.re_rank_and_persist()
    print("[Startup] FinVoice Autonomous Agent initialized and queue ranked.")

@app.get("/")
def health_check():
    return {
        "status": "ONLINE",
        "agent": settings.APP_NAME,
        "compliance": "RBI_HOURS_ENABLED (08:00 - 19:00)",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/queue", response_model=QueueResponse)
def get_prioritized_queue(capacity: int = 60):
    """
    Returns the real-time ranked calling queue with manager explanations and compliance disclosures.
    """
    ranking_data = monitor.re_rank_and_persist(capacity=capacity)
    return QueueResponse(
        capacity=capacity,
        total_evaluated=ranking_data["total_accounts_evaluated"],
        eligible_count=ranking_data["eligible_for_contact"],
        deferred_count=ranking_data["deferred_count"],
        summary_metrics=ranking_data["summary_metrics"],
        queue=ranking_data["ranked_queue"]
    )

@app.post("/api/webhooks/cashfree")
async def cashfree_payment_webhook(payload: CashfreeWebhookPayload):
    """
    Receives incoming Cashfree payment events (full or partial payments).
    Updates ledger, marks promises fulfilled, and instantly re-ranks the queue.
    """
    parsed = cashfree_client.parse_payment_webhook(payload.dict())
    loan_id = parsed.get("loan_id")
    if not loan_id:
        raise HTTPException(status_code=400, detail="loan_id must be provided or identifiable in order_id")
        
    result = monitor.handle_payment_event(
        loan_id=loan_id,
        amount=parsed["amount"],
        order_id=parsed["order_id"],
        payment_method=parsed["payment_method"]
    )
    return result

@app.post("/api/signals/intent")
async def ingest_sarvam_signal(signal: SarvamSignalRequest):
    """
    Receives post-call speech transcripts (Saaras) or OCR notes (Vision) from teammates.
    Extracts intent, updates borrower state, triggers immediate action (e.g. Cashfree link),
    and instantly re-ranks the queue.
    """
    if signal.raw_text and not signal.detected_intent:
        # Auto-parse conversational transcript using Sarvam adapter
        parsed_intent = sarvam_adapter.parse_speech_or_ocr_transcript(signal.raw_text)
    else:
        parsed_intent = {
            "raw_text": signal.raw_text or "",
            "detected_intent": signal.detected_intent or "UNKNOWN",
            "extracted_amount": signal.extracted_amount or 0.0,
            "extracted_promise_date": signal.extracted_promise_date,
            "source": signal.source
        }
        
    result = monitor.handle_intent_signal(loan_id=signal.loan_id, intent_data=parsed_intent)
    return result

@app.post("/api/payments/create-link")
async def create_payment_link(req: CreatePaymentLinkRequest):
    """
    Generates an instant Cashfree payment link for a specific borrower/loan.
    """
    from database.connection import SessionLocal
    from database.models import Loan
    db = SessionLocal()
    try:
        loan = db.query(Loan).filter(Loan.id == req.loan_id).first()
        if not loan:
            raise HTTPException(status_code=404, detail=f"Loan {req.loan_id} not found")
        b = loan.borrower
        link_data = cashfree_client.create_payment_link(
            loan_id=loan.id,
            borrower_name=b.name if b else "Borrower",
            phone=b.phone if b else "9876543210",
            amount=req.amount,
            purpose=req.purpose
        )
        return link_data
    finally:
        db.close()

@app.get("/api/audit/fairness")
def get_regulatory_fairness_audit():
    """
    Runs full compliance audit: disparate impact ratio, zero proxy leakage, and permutation tests.
    """
    audit_report = auditor.run_full_audit()
    return audit_report

@app.post("/api/cron/check-promises")
def trigger_lapsed_promise_check():
    """
    Checks for broken promises whose date has lapsed and applies penalties.
    """
    result = monitor.check_lapsed_promises()
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
