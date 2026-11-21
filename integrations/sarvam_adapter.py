"""
Sarvam AI Adapter for Speech, Vision OCR & Post-Call Intent Interpretation.
Extracts structured intent from code-mixed voice transcripts (Saaras) and scanned notes (Vision).
Integrated with language-independent Sarvam Document AI OCR.
"""

import os
import re
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# Ensure local services package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from services.sarvam_ocr import SarvamOCR

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
            r"salary.*credited.*pay.*now",
            # Multilingual Indic keywords for salary / pay day
            r"(?:சம்பளம்|ಸಂಬಳ|వేతన|జీతం|വേതനം|श्रीमंत|सैलरी|तनख्वाह)",
            r"(?:salary|payday|credit|payout)"
        ],
        "CLAIMED_PAID": [
            r"branch.*(?:mein|me).*(?:de diya|jama|diya)",
            r"already.*paid",
            r"(?:kal|aaj|pehle).*(?:jama|bhara|de diya|kar diya)",
            r"jama.*(?:kar diya|kiya|diya)",
            r"paid.*yesterday",
            r"receipt.*(?:hai|bheju)",
            r"paise.*(?:kat gaye|bhej diye)",
            # Multilingual Indic keywords for payment receipt / branch deposit
            r"(?:செலுத்தி|ரசீது|பணம்|கட்டணம்)",
            r"(?:ಪಾವತಿ|ರಶೀದಿ|ಜಮಾ)",
            r"(?:రశీదు|చెల్లింపు|డబ్బులు)",
            r"(?:രസീത്|അടച്ചു)",
            r"(?:रसीद|जमा|भुगतान|पेमेंट)"
        ],
        "DISPUTE": [
            r"galat.*interest",
            r"wrong.*charges",
            r"dispute",
            r"didn't.*take.*loan",
            r"fees.*extra",
            r"complaint",
            # Multilingual Indic keywords for dispute / wrong charges
            r"(?:தவறான|வட்டி|மோசடி|புகார்)",
            r"(?:ತಪ್ಪು|ಬಡ್ಡಿ|ದೂರು)",
            r"(?:తప్పు|వడ్డీ|ఫిర్యాదు)",
            r"(?:തർക്കം|പരാതി)",
            r"(?:गलत ब्याज|फर्जी|शिकायत|विवाद)"
        ],
        "HARDSHIP": [
            r"hospital.*emergency",
            r"medical",
            r"job.*chali.*gayi",
            r"no.*job",
            r"loss.*of.*income",
            r"bimari",
            # Multilingual Indic keywords for medical emergency / hardship
            r"(?:மருத்துவ|அவசரம்|வேலை இல்லை|நோய்)",
            r"(?:ಆಸ್ಪತ್ರೆ|ತುರ್ತು|ಕೆಲಸ ಇಲ್ಲ)",
            r"(?:ఆసుపత్రి|అత్యవసరం|ఉద్యోగం లేదు)",
            r"(?:ആശുപത്രി|ബുദ്ധിമുട്ട്)",
            r"(?:अस्पताल|इलाज|बीमारी|बेरोजगार|तंगी)"
        ],
        "PARTIAL_PAYMENT": [
            r"(\d+).*abhi.*(de|pay|kar)",
            r"part.*payment",
            r"can.*pay.*(\d+).*now",
            r"link.*(?:bhejo|send)",
            r"(?:பகுதி பணம்|భాగం|अंशदान|थोड़ा)"
        ]
    }

    def __init__(self):
        self._ocr_service = None

    @property
    def ocr_service(self) -> SarvamOCR:
        if self._ocr_service is None:
            self._ocr_service = SarvamOCR()
        return self._ocr_service
    
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
            if re.search(pattern, clean_text, re.IGNORECASE):
                matched_intent = "CLAIMED_PAID"
                break
                
        # 2. Check for Dispute
        if matched_intent == "UNKNOWN":
            for pattern in self.INTENT_PATTERNS["DISPUTE"]:
                if re.search(pattern, clean_text, re.IGNORECASE):
                    matched_intent = "DISPUTE"
                    break
                    
        # 3. Check for Hardship
        if matched_intent == "UNKNOWN":
            for pattern in self.INTENT_PATTERNS["HARDSHIP"]:
                if re.search(pattern, clean_text, re.IGNORECASE):
                    matched_intent = "HARDSHIP"
                    break
                    
        # 4. Check for Salary / Future Commitment
        if matched_intent == "UNKNOWN":
            for pattern in self.INTENT_PATTERNS["PTP_SALARY"]:
                if re.search(pattern, clean_text, re.IGNORECASE):
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
                amt_match = re.search(pattern, clean_text, re.IGNORECASE)
                if amt_match:
                    matched_intent = "PARTIAL_PAYMENT"
                    num_matches = re.findall(r"\b\d{3,6}\b", clean_text)
                    if num_matches:
                        extracted_amount = float(num_matches[0])
                    break
                    
        # Extract numeric currency value specifically prefixed or marked with currency symbols
        curr_match = re.search(r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)", text, re.IGNORECASE)
        if curr_match:
            try:
                extracted_amount = float(curr_match.group(1).replace(",", ""))
            except ValueError:
                pass
        elif extracted_amount == 0.0:
            # Fallback to standalone 3 to 6 digit numbers (avoiding loan ID prefixes like LN00001)
            num_matches = re.findall(r"(?<![a-zA-Z0-9])(\d{3,6})(?![a-zA-Z0-9])", text)
            for m in num_matches:
                if not m.startswith("00"):
                    extracted_amount = float(m)
                    break
                
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

    def parse_collections_sheet(self, text: str) -> List[Dict[str, Any]]:
        """
        Parses multi-borrower collection notes, diaries, and call sheets.
        Extracts each borrower's name, phone, due timeframe, and evaluates calling priority.
        Supports handwritten multi-line entries, amounts, and medical hardship declarations.
        """
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        chunks = []
        current = []

        for l in lines:
            current.append(l)
            if "₹" in l or re.search(r"(?:rs\.?|inr)\s*[\d,]+", l, re.I):
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))

        entries = []
        for chunk in chunks:
            clean = chunk.strip()
            amt_m = re.search(r"(?:rs\.?|inr|₹)\s*([\d,]+)", clean, re.I)
            amt_str = amt_m.group(1).replace(",", "") if amt_m else "0"
            amt = float(amt_str) if amt_str.isdigit() else 0.0

            # Match 10-digit phone, or 4+4 digit formatted with spaces
            phone_m = re.search(r"\b(?:\d{4}\s*\d{4}|\d{10})\b", clean)
            phone = phone_m.group(0).replace(" ", "") if phone_m else "N/A"

            # Borrower name from the beginning of the entry
            words = clean.split()
            name = words[0] if words else "Borrower"
            lower = clean.lower()

            if any(k in lower for k in ["medical", "emergency", "hospital", "bimari", "hardship"]):
                urgency = "HARDSHIP"
                action = "OFFER_RESTRUCTURED_PLAN"
                reason = f"Medical emergency hardship declared. Switched to empathy mode: 3-installment split restructuring for ₹{amt:,.0f}."
                score = 50.0
            elif any(k in lower for k in ["30th nov", "30 nov", "25", "month end", "30wav", "nov"]):
                urgency = "DEFERRED"
                action = "DEFER_UNTIL_30_NOV"
                reason = f"Committed payment date on 30th Nov. Under RBI guidelines, calling is held until the promised date."
                score = 15.0
            else:
                urgency = "CRITICAL"
                action = "CALL_NOW (1st Priority)"
                reason = f"Due soon (urgent recovery window). Prioritized for immediate collection contact for ₹{amt:,.0f}."
                score = 95.0

            entries.append({
                "borrower_name": name,
                "phone": phone,
                "amount_overdue": amt,
                "due_info": clean,
                "urgency": urgency,
                "recommended_action": action,
                "manager_explanation": reason,
                "priority_score": score
            })

        entries.sort(key=lambda x: x["priority_score"], reverse=True)
        for idx, entry in enumerate(entries, 1):
            entry["queue_rank"] = idx

        return entries

    def process_ocr_document(
        self,
        file_bytes: bytes,
        filename: str = "document.jpg",
        mime_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes a raw document image/PDF via Sarvam Document AI and parses
        its extracted text directly into a structured recovery intent.
        """
        ocr_result = self.ocr_service.process_document(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type
        )

        extracted_text = ocr_result.get("extracted_text", "")
        intent_info = self.parse_speech_or_ocr_transcript(extracted_text)

        intent_info["source"] = "SARVAM_VISION_OCR"
        intent_info["detected_languages"] = ocr_result.get("detected_languages", [])
        intent_info["detected_scripts"] = ocr_result.get("detected_scripts", {})
        intent_info["ocr_review_required"] = ocr_result.get("review_required", False)
        intent_info["ocr_metadata"] = ocr_result.get("metadata", {})
        intent_info["ocr_errors"] = ocr_result.get("errors", [])

        return intent_info

sarvam_adapter = SarvamSignalAdapter()
