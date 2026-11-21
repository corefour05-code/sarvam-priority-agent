"""
Synthetic Data Generator for FinVoice Collections Hackathon (Problem 2).
Generates 4,200 realistic overdue accounts reflecting Meena's regional portfolio.
Includes hidden ground-truth outcomes to benchmark algorithm lift and test fairness.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from database.connection import init_db, SessionLocal
from database.models import Borrower, Loan, SignalEvent

random.seed(42)
np.random.seed(42)

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Reyansh", "Muhammad", "Sai", "Ayaan", "Krishna",
    "Ishaan", "Shaurya", "Rohan", "Atharva", "Kabir", "Diya", "Saanvi", "Aanya", "Aadhya", "Pari",
    "Ananya", "Fatima", "Aarohi", "Meera", "Myra", "Rhea", "Pooja", "Sunita", "Deepak", "Rajesh",
    "Suresh", "Ramesh", "Priya", "Kavita", "Anil", "Amit", "Manish", "Sunil", "Vikram", "Neha"
]

LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Singh", "Kumar", "Ali", "Khan", "Das", "Reddy", "Nair",
    "Iyer", "Rao", "Joshi", "Deshmukh", "Gupta", "Mehta", "Bose", "Choudhury", "Pillai", "Yadav"
]

LANGUAGES = ["hi", "ta", "te", "mr", "kn", "en"]
GENDERS = ["female", "male", "other"]
RELIGIONS = ["hindu", "muslim", "christian", "sikh", "jain", "other"]
CASTES = ["general", "obc", "sc", "st"]
PINCODES = ["110001", "400001", "560001", "600001", "700001", "500001", "380001", "208001"]

def generate_dataset(num_records: int = 4200):
    borrowers_data = []
    loans_data = []
    today = datetime.now()
    
    for i in range(1, num_records + 1):
        borrower_id = f"B{i:05d}"
        loan_id = f"LN{i:05d}"
        
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        name = f"{first} {last}"
        phone = f"+9198{random.randint(10000000, 99999999)}"
        lang = random.choice(LANGUAGES)
        
        # Demographic attributes for compliance fairness audit
        gender = random.choice(GENDERS)
        religion = random.choice(RELIGIONS)
        caste = random.choice(CASTES)
        pincode = random.choice(PINCODES)
        
        # Loan financial characteristics
        principal = round(random.choice([15000, 25000, 50000, 80000, 120000, 200000, 300000]), 2)
        total_installments = random.choice([6, 12, 18, 24, 36])
        installments_paid = random.randint(0, total_installments - 1)
        
        # Overdue parameters
        amount_overdue = round(min(principal, (principal / total_installments) * random.randint(1, 4)), 2)
        dpd = random.randint(1, 120)
        
        # Payment behavior history
        days_since_last_payment = random.randint(3, 120) if installments_paid > 0 else 999
        last_payment_date = today - timedelta(days=days_since_last_payment) if installments_paid > 0 else None
        last_payment_amount = round(amount_overdue * random.uniform(0.5, 1.0), 2) if installments_paid > 0 else 0.0
        
        # Commitment & Promise History
        promises_made = random.randint(0, 5)
        promises_fulfilled = random.randint(0, promises_made) if promises_made > 0 else 0
        
        # Contact Reachability History
        contact_attempts = random.randint(0, 6)
        contact_responses = random.randint(0, contact_attempts) if contact_attempts > 0 else 0
        
        # Status assignment (distribution of realistic portfolio issues)
        status_roll = random.random()
        active_promise_date = None
        active_promise_amount = 0.0
        promise_status = "NONE"
        dispute_reason = None
        hardship_reason = None
        
        if status_roll < 0.04:
            # 4% Already Paid claims ("branch mein kal jama kiya")
            status = "PENDING_VERIFICATION"
        elif status_roll < 0.07:
            # 3% Active dispute
            status = "DISPUTED"
            dispute_reason = "Borrower disputes interest calculation and hidden processing fees"
        elif status_roll < 0.10:
            # 3% Hardship case
            status = "HARDSHIP"
            hardship_reason = "Loss of primary family income due to medical emergency"
        elif status_roll < 0.25:
            # 15% Active Promise to Pay (PTP)
            status = "ACTIVE"
            promise_offset = random.randint(-5, 10) # negative means lapsed, positive means future
            ptp_date = today + timedelta(days=promise_offset)
            active_promise_date = ptp_date
            active_promise_amount = amount_overdue
            if promise_offset > 0:
                promise_status = "PENDING"
            else:
                promise_status = "BROKEN"
        else:
            status = "ACTIVE"
            
        # Underlying Ground Truth (True payment probability for benchmarking)
        # Note: STRICTLY dependent on financial & behavioral features, ZERO demographic correlation!
        fulfillment_rate = (promises_fulfilled / promises_made) if promises_made > 0 else 0.5
        repay_rate = (installments_paid / total_installments)
        dpd_factor = max(0.0, 1.0 - (dpd / 120.0))
        recency_factor = max(0.0, 1.0 - (days_since_last_payment / 90.0)) if days_since_last_payment < 999 else 0.1
        response_rate = (contact_responses / contact_attempts) if contact_attempts > 0 else 0.4
        
        base_pay_logit = (
            2.0 * fulfillment_rate +
            1.5 * repay_rate +
            1.8 * dpd_factor +
            1.2 * recency_factor -
            (2.5 if status == "DISPUTED" else 0.0) -
            (1.5 if status == "HARDSHIP" else 0.0) -
            (1.0 if promise_status == "BROKEN" else 0.0) +
            (1.5 if promise_status == "PENDING" and promise_offset == 0 else 0.0) -
            1.8
        )
        true_prob_pay = 1.0 / (1.0 + np.exp(-base_pay_logit))
        
        # Ground truth reachability
        true_prob_contact = 0.35 + 0.55 * response_rate
        
        # Stored true outcome if contacted
        true_will_pay = (random.random() < true_prob_pay) if status not in ["DISPUTED", "PENDING_VERIFICATION"] else False
        true_contact_answers = (random.random() < true_prob_contact)
        
        borrowers_data.append({
            "id": borrower_id,
            "name": name,
            "phone": phone,
            "preferred_language": lang,
            "gender": gender,
            "religion": religion,
            "caste_category": caste,
            "pincode": pincode,
            "created_at": today
        })
        
        loans_data.append({
            "id": loan_id,
            "borrower_id": borrower_id,
            "principal_amount": principal,
            "amount_overdue": amount_overdue,
            "days_past_due": dpd,
            "installments_total": total_installments,
            "installments_paid": installments_paid,
            "last_payment_date": last_payment_date,
            "last_payment_amount": last_payment_amount,
            "status": status,
            "dispute_reason": dispute_reason,
            "hardship_reason": hardship_reason,
            "active_promise_date": active_promise_date,
            "active_promise_amount": active_promise_amount,
            "promise_status": promise_status,
            "promises_made_count": promises_made,
            "promises_fulfilled_count": promises_fulfilled,
            "contact_attempts_count": contact_attempts,
            "contact_responses_count": contact_responses,
            "last_contact_outcome": "ANSWERED" if contact_responses > 0 else "NO_RESPONSE",
            # Ground truth columns (used ONLY for benchmark verification, never during feature input)
            "true_prob_pay": round(float(true_prob_pay), 4),
            "true_prob_contact": round(float(true_prob_contact), 4),
            "true_will_pay": true_will_pay,
            "true_contact_answers": true_contact_answers
        })
        
    df_borrowers = pd.DataFrame(borrowers_data)
    df_loans = pd.DataFrame(loans_data)
    df_combined = pd.merge(df_loans, df_borrowers, left_on="borrower_id", right_on="id", suffixes=("", "_borrower"))
    
    # Save CSV artifact for curveball and offline analysis
    os.makedirs("./data", exist_ok=True)
    df_combined.to_csv("./data/synthetic_borrowers_4200.csv", index=False)
    print(f"Generated {len(df_combined)} synthetic accounts in ./data/synthetic_borrowers_4200.csv")
    
    return df_borrowers, df_loans

def seed_database():
    init_db()
    db = SessionLocal()
    
    # Check if already seeded
    existing_loans = db.query(Loan).count()
    if existing_loans >= 4200:
        print(f"Database already contains {existing_loans} loans. Skipping seed.")
        db.close()
        return
        
    print("Seeding database with 4,200 accounts...")
    df_borrowers, df_loans = generate_dataset(4200)
    
    # Drop existing borrowers and loans to have a fresh state
    db.query(Loan).delete()
    db.query(Borrower).delete()
    db.commit()
    
    borrower_records = df_borrowers.replace({np.nan: None}).to_dict(orient="records")
    loan_records = df_loans.replace({np.nan: None}).to_dict(orient="records")
    
    borrower_objs = []
    for r in borrower_records:
        b = Borrower(
            id=r["id"],
            name=r["name"],
            phone=r["phone"],
            preferred_language=r["preferred_language"],
            gender=r["gender"],
            religion=r["religion"],
            caste_category=r["caste_category"],
            pincode=r["pincode"],
            created_at=r["created_at"].to_pydatetime() if hasattr(r["created_at"], "to_pydatetime") else r["created_at"]
        )
        borrower_objs.append(b)
        
    db.bulk_save_objects(borrower_objs)
    db.commit()
    
    loan_objs = []
    for r in loan_records:
        last_pay_dt = r["last_payment_date"]
        if last_pay_dt is not None and hasattr(last_pay_dt, "to_pydatetime"):
            last_pay_dt = last_pay_dt.to_pydatetime()
        elif pd.isna(last_pay_dt):
            last_pay_dt = None
            
        ptp_dt = r["active_promise_date"]
        if ptp_dt is not None and hasattr(ptp_dt, "to_pydatetime"):
            ptp_dt = ptp_dt.to_pydatetime()
        elif pd.isna(ptp_dt):
            ptp_dt = None

        l = Loan(
            id=r["id"],
            borrower_id=r["borrower_id"],
            principal_amount=float(r["principal_amount"]),
            amount_overdue=float(r["amount_overdue"]),
            days_past_due=int(r["days_past_due"]),
            installments_total=int(r["installments_total"]),
            installments_paid=int(r["installments_paid"]),
            last_payment_date=last_pay_dt,
            last_payment_amount=float(r["last_payment_amount"]),
            status=str(r["status"]),
            dispute_reason=str(r["dispute_reason"]) if r["dispute_reason"] is not None and not pd.isna(r["dispute_reason"]) else None,
            hardship_reason=str(r["hardship_reason"]) if r["hardship_reason"] is not None and not pd.isna(r["hardship_reason"]) else None,
            active_promise_date=ptp_dt,
            active_promise_amount=float(r["active_promise_amount"]),
            promise_status=str(r["promise_status"]),
            promises_made_count=int(r["promises_made_count"]),
            promises_fulfilled_count=int(r["promises_fulfilled_count"]),
            contact_attempts_count=int(r["contact_attempts_count"]),
            contact_responses_count=int(r["contact_responses_count"]),
            last_contact_outcome=str(r["last_contact_outcome"])
        )
        loan_objs.append(l)
        
    db.bulk_save_objects(loan_objs)
    db.commit()
    print("Database seeding completed successfully: 4,200 accounts loaded.")
    db.close()

if __name__ == "__main__":
    seed_database()
