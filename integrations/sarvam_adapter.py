"""
Sarvam AI Adapter for Speech, Vision OCR & Post-Call Intent Interpretation.
Extracts structured intent from code-mixed voice transcripts (Saaras) and scanned notes (Vision).
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class SarvamSignalAdapter:
    """
    Interprets messy code-mixed borrower speech and handwritten notes
    into clean, normalized state changes.
    """
    
    INTENT_PATTERNS = {
        "PTP_SALARY": [
            r"salary.*(?:aayega|milega|credit|5th|10th|25th|aane do)",
            r"tankha.*(?:milegi|aayegi)",
            r"pay on.*(?:5th|10th|25th|month end)",
            r"salary.*credited.*pay.*now"
        ],
        "CLAIMED_PAID": [
            r"branch.*(?:mein|me).*(?:de diya|jama|diya)",
            r"already.*paid",
            r"(?:kal|aaj|pehle).*(?:jama|bhara|de diya|kar diya)",
            r"jama.*(?:kar diya|kiya|diya)",
            r"paid.*yesterday",
            r"receipt.*(?:hai|bheju)",
            r"paise.*(?:kat gaye|bhej diye)"
        ],
        "DISPUTE": [
            r"galat.*interest",
            r"wrong.*charges",
            r"dispute",
            r"didn't.*take.*loan",
            r"fees.*extra",
            r"complaint"
        ],
        "HARDSHIP": [
            r"hospital.*emergency",
            r"medical",
            r"job.*chali.*gayi",
            r"no.*job",
            r"loss.*of.*income",
            r"bimari"
        ],
        "PARTIAL_PAYMENT": [
            r"(\d+).*abhi.*(de|pay|kar)",
            r"part.*payment",
            r"can.*pay.*(\d+).*now",
            r"link.*(?:bhejo|send)"
        ]
    }
    
    def parse_speech_or_ocr_transcript(self, text: str, current_dt: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Parses raw conversational text into normalized intent, dates, and amounts.
        """
        if current_dt is None:
            current_dt = datetime.now()
            
        clean_text = text.lower().strip()
        matched_intent = "UNKNOWN"
        extracted_amount = 0.0
        extracted_date = None
        
        # 1. Check for Claimed Paid
        for pattern in self.INTENT_PATTERNS["CLAIMED_PAID"]:
            if re.search(pattern, clean_text):
                matched_intent = "CLAIMED_PAID"
                break
                
        # 2. Check for Dispute
        if matched_intent == "UNKNOWN":
            for pattern in self.INTENT_PATTERNS["DISPUTE"]:
                if re.search(pattern, clean_text):
                    matched_intent = "DISPUTE"
                    break
                    
        # 3. Check for Hardship
        if matched_intent == "UNKNOWN":
            for pattern in self.INTENT_PATTERNS["HARDSHIP"]:
                if re.search(pattern, clean_text):
                    matched_intent = "HARDSHIP"
                    break
                    
        # 4. Check for Salary / Future Commitment
        if matched_intent == "UNKNOWN":
            for pattern in self.INTENT_PATTERNS["PTP_SALARY"]:
                if re.search(pattern, clean_text):
                    matched_intent = "PTP_SALARY"
                    # Try to extract specific date e.g. 5th, 10th, 25th
                    day_match = re.search(r"(\d{1,2})(?:th|st|nd|rd)?", clean_text)
                    if day_match:
                        day_num = int(day_match.group(1))
                        if 1 <= day_num <= 31:
                            target_year = current_dt.year
                            target_month = current_dt.month
                            if day_num <= current_dt.day:
                                # Target is in next month
                                target_month = 12 if current_dt.month == 12 else current_dt.month + 1
                                if target_month == 1:
                                    target_year += 1
                            try:
                                extracted_date = datetime(target_year, target_month, day_num)
                            except ValueError:
                                extracted_date = current_dt + timedelta(days=5)
                    if extracted_date is None:
                        extracted_date = current_dt + timedelta(days=5)
                    break
                    
        # 5. Check for Partial Payment / Immediate Link Request
        if matched_intent in ["UNKNOWN", "PTP_SALARY"]:
            for pattern in self.INTENT_PATTERNS["PARTIAL_PAYMENT"]:
                amt_match = re.search(pattern, clean_text)
                if amt_match:
                    matched_intent = "PARTIAL_PAYMENT"
                    # Try to extract amount
                    num_matches = re.findall(r"\b\d{3,6}\b", clean_text)
                    if num_matches:
                        extracted_amount = float(num_matches[0])
                    break
                    
        # Extract any numeric currency value
        if extracted_amount == 0.0:
            amt_find = re.findall(r"(?:rs\.?|inr|₹)?\s*(\d{3,6})", clean_text)
            if amt_find:
                extracted_amount = float(amt_find[0])
                
        return {
            "raw_text": text,
            "detected_intent": matched_intent,
            "extracted_amount": extracted_amount,
            "extracted_promise_date": extracted_date.isoformat() if extracted_date else None,
            "confidence": 0.92 if matched_intent != "UNKNOWN" else 0.35,
            "recommended_action": {
                "CLAIMED_PAID": "PAUSE_CALLS_VERIFY_BRANCH",
                "DISPUTE": "FREEZE_AND_ROUTE_GRIEVANCE",
                "HARDSHIP": "OFFER_RESTRUCTURED_PLAN",
                "PTP_SALARY": "SCHEDULE_SALARY_DATE_FOLLOWUP",
                "PARTIAL_PAYMENT": "SEND_CASHFREE_LINK_NOW",
                "UNKNOWN": "STANDARD_FOLLOWUP"
            }.get(matched_intent, "STANDARD_FOLLOWUP")
        }

sarvam_adapter = SarvamSignalAdapter()
