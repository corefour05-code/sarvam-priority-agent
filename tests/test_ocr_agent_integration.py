"""
End-to-End Verification: Sarvam OCR -> Priority Agent Ingestion
Tests uploading a scanned document, running Sarvam Document AI OCR,
extracting borrower intent, and re-ranking the Priority Agent calling queue.
"""

import sys
import os
import io
from PIL import Image, ImageDraw, ImageFont

# Add project roots to sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("Priority Agent/priorityAgent2"))

from integrations.sarvam_adapter import sarvam_adapter
from engine.monitor import monitor
from database.connection import init_db, SessionLocal
from database.models import Loan

init_db()

def create_sample_receipt_image() -> bytes:
    """Creates a sample branch deposit receipt with mixed Tamil + English text."""
    width = 750
    height = 240
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/Nirmala.ttc", 26, index=0)

    lines = [
        "BANK DEPOSIT RECEIPT / வங்கி ரசீது",
        "Loan ID: LN00001 - Borrower: Sai Sharma",
        "Amount: ₹ 5,000 paid at branch counter",
        "Status: Already paid / ஏற்கனவே செலுத்திவிட்டேன்"
    ]

    y = 30
    for line in lines:
        draw.text((35, y), line, fill="black", font=font)
        y += 50

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

def run_integration_test():
    print("================================================================================")
    print("TESTING OCR -> PRIORITY AGENT PIPELINE INTEGRATION")
    print("================================================================================")

    # 1. Check initial loan state
    db = SessionLocal()
    target_loan = db.query(Loan).filter(Loan.id == "LN00001").first()
    print(f"\n[Initial Loan State] Loan: {target_loan.id}, Status: {target_loan.status}, Overdue: ₹{target_loan.amount_overdue}")
    db.close()

    # 2. Generate and process OCR document
    print("\n[Step 1] Running Sarvam Document AI OCR on Scanned Document...")
    img_bytes = create_sample_receipt_image()
    
    intent_data = sarvam_adapter.process_ocr_document(
        file_bytes=img_bytes,
        filename="bank_receipt_LN00001.jpg"
    )

    print("\n[Step 2] OCR Extraction & Intent Results:")
    print(f"  Extracted Text:\n{intent_data['raw_text']}")
    print(f"  Detected Intent: {intent_data['detected_intent']}")
    print(f"  Extracted Amount: ₹{intent_data['extracted_amount']:,.0f}")
    print(f"  Detected Languages: {intent_data['detected_languages']}")
    print(f"  Recommended Action: {intent_data['recommended_action']}")

    # 3. Dispatch to Priority Agent State Machine
    print("\n[Step 3] Dispatching Signal to FinVoice Priority Agent Monitor...")
    agent_result = monitor.handle_intent_signal(
        loan_id="LN00001",
        intent_data=intent_data
    )

    print("\n[Step 4] Priority Agent Decision & Replanning:")
    print(f"  Success: {agent_result['success']}")
    print(f"  Action Taken: {agent_result['action_taken']}")
    
    # 4. Verify updated queue
    print("\n[Step 5] Dynamically Re-ranked Top 5 Calling Queue:")
    for q in agent_result["top_5_queue"][:5]:
        print(f"  Rank #{q['queue_rank']}: {q['borrower_name']} ({q['loan_id']}) | Overdue: ₹{q['amount_overdue']:,.0f} | Action: {q['recommended_action']}")

    # 5. Verify loan status updated
    db = SessionLocal()
    updated_loan = db.query(Loan).filter(Loan.id == "LN00001").first()
    print(f"\n[Final Loan State] Loan: {updated_loan.id}, New Status: {updated_loan.status}")
    db.close()

    assert agent_result["success"] is True
    print("\n================================================================================")
    print("✓ OCR -> PRIORITY AGENT PIPELINE INTEGRATION VERIFIED SUCCESSFULLY!")
    print("================================================================================")

if __name__ == "__main__":
    run_integration_test()
