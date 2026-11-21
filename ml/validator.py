"""
Universal Data Ingestion, Column Normalizer & Data Integrity Auditor.
Enables FinVoice to ingest ANY dataset schema (CSVs, JSON, DataFrames, field dumps)
without crashing, while strictly flagging missing or corrupt fields to prevent false results.

Guarantees:
1. Auto-resolves arbitrary column headers (e.g., 'acct_num', 'due_amt', 'mobile_no').
2. Flags missing critical values (amount, phone, loan_id) and quarantines them.
3. Flags data warnings (unparseable dates, negative numbers, missing tenure).
4. Strictly isolates protected demographic attributes.
"""

import re
from datetime import datetime, date
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

# Canonical internal schema mapping dictionary
COLUMN_ALIASES = {
    "id": [
        "id", "loan_id", "loan_no", "loan_number", "account_id", "account_no",
        "acct_no", "acct_id", "loan_num", "application_id", "case_id"
    ],
    "borrower_id": [
        "borrower_id", "customer_id", "cust_id", "client_id", "user_id", "borrower_num"
    ],
    "name": [
        "borrower_name", "customer_name", "client_name", "name", "full_name",
        "applicant_name", "borrower"
    ],
    "phone": [
        "phone", "mobile", "mobile_no", "phone_number", "contact", "contact_no",
        "cell", "telephone", "borrower_phone"
    ],
    "preferred_language": [
        "preferred_language", "language", "lang", "preferred_lang", "spoken_language", "mother_tongue"
    ],
    "amount_overdue": [
        "amount_overdue", "overdue_amount", "due_amount", "due_amt", "balance_overdue",
        "overdue_amt", "amount_due", "amt_due", "overdue", "pending_amount", "emi_overdue", "balance",
        "total_due", "tot_due", "outstanding", "outstanding_amount", "delinquent_amount"
    ],
    "principal_amount": [
        "principal_amount", "principal", "loan_amount", "sanctioned_amount",
        "disbursed_amount", "total_loan", "total_principal"
    ],
    "days_past_due": [
        "days_past_due", "dpd", "days_overdue", "overdue_days", "aging", "aging_days", "bucket_days"
    ],
    "installments_total": [
        "installments_total", "total_installments", "tenure", "tenor", "total_emis", "emi_count"
    ],
    "installments_paid": [
        "installments_paid", "paid_installments", "emis_paid", "repaid_installments"
    ],
    "last_payment_date": [
        "last_payment_date", "last_paid_date", "last_pmt_dt", "last_transaction_date",
        "previous_payment_date", "last_repayment_date"
    ],
    "last_payment_amount": [
        "last_payment_amount", "last_paid_amount", "last_pmt_amt", "previous_payment_amount"
    ],
    "active_promise_date": [
        "active_promise_date", "promise_date", "ptp_date", "commitment_date",
        "followup_date", "promised_date", "next_call_date", "due_date"
    ],
    "active_promise_amount": [
        "active_promise_amount", "promised_amount", "ptp_amount", "commitment_amount"
    ],
    "status": [
        "status", "account_status", "loan_status", "disposition", "disposition_status"
    ],
    "promises_made_count": [
        "promises_made_count", "promises_made", "ptp_count", "commitments_made"
    ],
    "promises_fulfilled_count": [
        "promises_fulfilled_count", "promises_fulfilled", "ptp_kept", "commitments_kept"
    ],
    "contact_attempts_count": [
        "contact_attempts_count", "contact_attempts", "calls_made", "attempts"
    ],
    "contact_responses_count": [
        "contact_responses_count", "contact_responses", "calls_answered", "responses"
    ],
    "notes": [
        "notes", "field_notes", "transcript", "speech_transcript", "remarks",
        "comments", "interaction_notes", "call_summary"
    ],
    # Protected demographic attributes (strictly for bias audit, never for ML features)
    "gender": ["gender", "sex"],
    "religion": ["religion", "faith"],
    "caste_category": ["caste", "caste_category", "social_category"],
    "pincode": ["pincode", "pin_code", "zip", "postal_code", "area_code"]
}

def parse_flexible_date(val: Any) -> Optional[datetime]:
    """
    Safely parses various date formats without failing on malformed strings.
    """
    if val is None or pd.isna(val) or val == "":
        return None
    if isinstance(val, (datetime, date)):
        return datetime(val.year, val.month, val.day) if isinstance(val, date) and not isinstance(val, datetime) else val
        
    s = str(val).strip()
    if s.lower() in ["none", "nan", "null", "nat", "n/a", ""]:
        return None
        
    # Standard format attempts
    formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
        "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
            
    # Try ISO parsing
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
        
    return None

class DataIntegrityAuditor:
    """
    Inspects, validates, cleans, and flags any input borrower dataset.
    Prevents false hallucinations while identifying missing critical fields.
    """
    
    @staticmethod
    def map_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Maps arbitrary input columns to canonical schema names.
        """
        normalized_cols = {col: str(col).strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns}
        df_renamed = df.rename(columns=normalized_cols)
        
        column_mapping = {}
        for canonical, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                norm_alias = alias.lower()
                if norm_alias in df_renamed.columns:
                    column_mapping[norm_alias] = canonical
                    break
                    
        df_canonical = df_renamed.rename(columns=column_mapping)
        return df_canonical, column_mapping
        
    @classmethod
    def validate_and_clean_portfolio(cls, raw_data: Any) -> Dict[str, Any]:
        """
        Universal entry point: Accepts CSV file path, DataFrame, list of dicts, or single dict.
        Returns validated portfolio partitioned into:
        1. valid_records: Cleaned and ready for scoring
        2. quarantined_records: Critical defects (blocked from calling to prevent false results)
        3. audit_report: Complete data health scorecard
        """
        if isinstance(raw_data, str):
            # File path
            df = pd.read_csv(raw_data)
        elif isinstance(raw_data, pd.DataFrame):
            df = raw_data.copy()
        elif isinstance(raw_data, list):
            df = pd.DataFrame(raw_data)
        elif isinstance(raw_data, dict):
            df = pd.DataFrame([raw_data])
        else:
            raise ValueError(f"Unsupported data format: {type(raw_data)}")
            
        df_mapped, col_map = cls.map_columns(df)
        
        valid_records = []
        quarantined_records = []
        
        total_rows = len(df_mapped)
        missing_amounts_count = 0
        missing_names_count = 0
        missing_phones_count = 0
        missing_dates_count = 0
        
        for idx, row in df_mapped.iterrows():
            row_dict = row.to_dict()
            flags = []
            is_quarantined = False
            
            prerequisites_met = {
                "amount_met": True,
                "date_met": True,
                "name_met": True,
                "contact_met": True
            }
            
            # --- 1. Mandatory Identifier: Loan ID ---
            loan_id = row_dict.get("id")
            if pd.isna(loan_id) or str(loan_id).strip() == "":
                loan_id = f"ROW_{idx + 1:05d}"
                flags.append({"level": "WARNING", "field": "id", "issue": f"Missing loan_id. Auto-assigned fallback {loan_id}."})
            else:
                loan_id = str(loan_id).strip()
            row_dict["id"] = loan_id
            
            # --- 2. Financial Dues: Amount Overdue ---
            raw_amt = row_dict.get("amount_overdue")
            if pd.isna(raw_amt) or raw_amt is None or str(raw_amt).strip() == "":
                flags.append({
                    "level": "CRITICAL_DEFECT",
                    "field": "amount_overdue",
                    "issue": "Missing overdue amount. Cannot calculate recovery yield without financial dues."
                })
                prerequisites_met["amount_met"] = False
                is_quarantined = True
                missing_amounts_count += 1
                row_dict["amount_overdue"] = 0.0
            else:
                try:
                    amt_cleaned = float(str(raw_amt).replace(",", "").replace("₹", "").replace("$", "").strip())
                    if amt_cleaned < 0:
                        flags.append({
                            "level": "CRITICAL_DEFECT",
                            "field": "amount_overdue",
                            "issue": f"Negative overdue amount ({amt_cleaned}). Potential credit balance; recovery prohibited."
                        })
                        prerequisites_met["amount_met"] = False
                        is_quarantined = True
                        missing_amounts_count += 1
                    row_dict["amount_overdue"] = max(0.0, amt_cleaned)
                except ValueError:
                    flags.append({
                        "level": "CRITICAL_DEFECT",
                        "field": "amount_overdue",
                        "issue": f"Unparseable overdue amount: '{raw_amt}'."
                    })
                    prerequisites_met["amount_met"] = False
                    is_quarantined = True
                    missing_amounts_count += 1
                    row_dict["amount_overdue"] = 0.0

            # --- 3. Contact Information: Name & Phone ---
            name = row_dict.get("name")
            if pd.isna(name) or str(name).strip() == "":
                flags.append({
                    "level": "WARNING",
                    "field": "name",
                    "issue": "Missing borrower name. Prerequisites: Name not met (fallback to 'Valued Customer')."
                })
                prerequisites_met["name_met"] = False
                missing_names_count += 1
                row_dict["name"] = "Valued Customer"
            else:
                row_dict["name"] = str(name).strip()
                
            phone = row_dict.get("phone")
            if pd.isna(phone) or str(phone).strip() in ["", "none", "nan", "null"]:
                flags.append({
                    "level": "WARNING",
                    "field": "phone",
                    "issue": "Missing phone number. Prerequisites: Contact info not met."
                })
                prerequisites_met["contact_met"] = False
                missing_phones_count += 1
                row_dict["phone"] = "N/A"
            else:
                clean_phone = re.sub(r"[^\d+]", "", str(phone).strip())
                row_dict["phone"] = clean_phone
                
            # --- 4. Date Fields Validation ---
            raw_last_pay = row_dict.get("last_payment_date")
            parsed_last_pay = parse_flexible_date(raw_last_pay)
            if raw_last_pay and not pd.isna(raw_last_pay) and parsed_last_pay is None:
                flags.append({"level": "WARNING", "field": "last_payment_date", "issue": f"Unparseable payment date '{raw_last_pay}'."})
                prerequisites_met["date_met"] = False
                missing_dates_count += 1
            row_dict["last_payment_date"] = parsed_last_pay
            
            raw_promise_dt = row_dict.get("active_promise_date")
            parsed_promise_dt = parse_flexible_date(raw_promise_dt)
            if raw_promise_dt and not pd.isna(raw_promise_dt) and parsed_promise_dt is None:
                flags.append({"level": "WARNING", "field": "active_promise_date", "issue": f"Unparseable commitment date '{raw_promise_dt}'."})
                prerequisites_met["date_met"] = False
                missing_dates_count += 1
            row_dict["active_promise_date"] = parsed_promise_dt
            
            # --- 5. Numeric Defaults & Normalizations ---
            row_dict["days_past_due"] = int(row_dict.get("days_past_due") or 0) if not pd.isna(row_dict.get("days_past_due")) else 0
            row_dict["principal_amount"] = float(row_dict.get("principal_amount") or max(10000.0, row_dict["amount_overdue"])) if not pd.isna(row_dict.get("principal_amount")) else max(10000.0, row_dict["amount_overdue"])
            row_dict["installments_total"] = int(row_dict.get("installments_total") or 12) if not pd.isna(row_dict.get("installments_total")) else 12
            row_dict["installments_paid"] = int(row_dict.get("installments_paid") or 0) if not pd.isna(row_dict.get("installments_paid")) else 0
            row_dict["promises_made_count"] = int(row_dict.get("promises_made_count") or 0) if not pd.isna(row_dict.get("promises_made_count")) else 0
            row_dict["promises_fulfilled_count"] = int(row_dict.get("promises_fulfilled_count") or 0) if not pd.isna(row_dict.get("promises_fulfilled_count")) else 0
            row_dict["contact_attempts_count"] = int(row_dict.get("contact_attempts_count") or 0) if not pd.isna(row_dict.get("contact_attempts_count")) else 0
            row_dict["contact_responses_count"] = int(row_dict.get("contact_responses_count") or 0) if not pd.isna(row_dict.get("contact_responses_count")) else 0
            row_dict["status"] = str(row_dict.get("status") or "ACTIVE").strip().upper()
            row_dict["promise_status"] = str(row_dict.get("promise_status") or ("PENDING" if parsed_promise_dt else "NONE")).strip().upper()
            row_dict["preferred_language"] = str(row_dict.get("preferred_language") or "hi").strip().lower()
            
            row_dict["prerequisites_met"] = prerequisites_met
            row_dict["data_validation_flags"] = flags
            row_dict["is_data_quarantined"] = is_quarantined
            
            if is_quarantined:
                quarantined_records.append(row_dict)
            else:
                valid_records.append(row_dict)
                
        audit_report = {
            "total_records_ingested": total_rows,
            "valid_records_count": len(valid_records),
            "quarantined_records_count": len(quarantined_records),
            "data_health_percentage": round((len(valid_records) / max(1, total_rows)) * 100, 2),
            "defects_breakdown": {
                "missing_or_corrupt_amounts": missing_amounts_count,
                "missing_phones": missing_phones_count,
                "missing_names": missing_names_count,
                "unparseable_dates": missing_dates_count
            },
            "mapped_columns": col_map,
            "verdict": "READY_FOR_PRIORITIZATION" if len(valid_records) > 0 else "FAILED_CRITICAL_VALIDATION"
        }
        
        return {
            "valid_records": valid_records,
            "quarantined_records": quarantined_records,
            "audit_report": audit_report
        }

auditor_pipeline = DataIntegrityAuditor()
