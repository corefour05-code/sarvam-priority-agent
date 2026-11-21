import json
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from services.sarvam_ocr import SarvamOCR, get_mime_type

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("SARVAM_API_KEY")

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="FinVoice OCR Lab",
    page_icon="📄",
    layout="wide",
)

st.title("FinVoice — Multilingual Document OCR")
st.caption(
    "Language-independent document extraction powered by Sarvam Document AI"
)

if not API_KEY:
    st.error(
        "SARVAM_API_KEY is missing. Please add it to your .env file."
    )
    st.stop()


@st.cache_resource
def get_ocr_service():
    return SarvamOCR(api_key=API_KEY)


ocr_service = get_ocr_service()


def clean_filename(filename: str) -> str:
    filename = Path(filename).name
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    if not Path(filename).suffix:
        filename += ".jpg"
    return filename


# ---------------------------------------------------------
# UI: DOCUMENT INPUT (NO LANGUAGE SELECTION)
# ---------------------------------------------------------

st.subheader("Upload a document")

input_method = st.radio(
    "Choose input method",
    ["Upload image/document", "Capture using camera"],
    horizontal=True,
)

document = None

if input_method == "Upload image/document":
    document = st.file_uploader(
        "Upload JPG, JPEG, PNG or PDF",
        type=["jpg", "jpeg", "png", "pdf"],
    )
else:
    document = st.camera_input(
        "Capture your document"
    )

# ---------------------------------------------------------
# PROCESS DOCUMENT
# ---------------------------------------------------------

if document is not None:
    file_bytes = document.getvalue()

    original_name = getattr(document, "name", "")
    if input_method == "Capture using camera":
        filename = "camera_capture.jpg"
    else:
        filename = clean_filename(original_name or "uploaded_document.jpg")

    st.image(
        file_bytes,
        caption="Uploaded document",
        use_container_width=True,
    )

    if len(file_bytes) > 20 * 1024 * 1024:
        st.error("File size exceeds maximum allowed limit of 20 MB.")
        st.stop()

    if st.button(
        "Run language-independent OCR",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Processing document with Sarvam Document AI..."):
                result = ocr_service.process_document(
                    file_bytes=file_bytes,
                    filename=filename,
                    mime_type=get_mime_type(filename),
                )

            st.success("OCR processing finished.")

            # -----------------------------------------
            # DETECTED LANGUAGES & STATUS
            # -----------------------------------------
            st.subheader("Detected Language & Status")

            languages_str = ", ".join(result["detected_languages"]) if result["detected_languages"] else "Unknown"
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"### {languages_str}")
                if result["review_required"]:
                    st.warning(
                        "⚠️ Review required: The extracted text contains minimal script evidence or encountered errors."
                    )
                else:
                    st.success(
                        "✅ Reliable script evidence detected."
                    )

            with col2:
                st.metric(
                    "Processing Time",
                    f"{result['metadata'].get('duration_seconds', 0)}s",
                )
                st.caption(
                    f"Characters analyzed: {result['metadata'].get('total_characters', 0)}"
                )

            # -----------------------------------------
            # DETECTED SCRIPTS TABLE
            # -----------------------------------------
            st.subheader("Detected Scripts")

            script_rows = []
            for script_name, details in result["detected_scripts"].items():
                script_rows.append({
                    "Script": script_name,
                    "Characters": details["characters"],
                    "Ratio": f"{round(details['ratio'] * 100, 1)}%",
                })

            if script_rows:
                st.dataframe(
                    script_rows,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No alphabetic script characters were detected.")

            # -----------------------------------------
            # EXTRACTED TEXT
            # -----------------------------------------
            st.subheader("Extracted Text")

            st.text_area(
                "OCR Extracted Content",
                value=result["extracted_text"],
                height=250,
            )

            st.download_button(
                "Download extracted text",
                data=result["extracted_text"],
                file_name=f"{Path(filename).stem}_extracted.txt",
                mime="text/plain",
            )

            # -----------------------------------------
            # ERRORS & WARNINGS (IF ANY)
            # -----------------------------------------
            if result.get("errors"):
                st.subheader("Errors & Warnings")
                st.json(result["errors"])

            # -----------------------------------------
            # STRUCTURED REPORT DOWNLOAD
            # -----------------------------------------
            st.download_button(
                "Download complete OCR report (JSON)",
                data=json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                ),
                file_name=f"{Path(filename).stem}_ocr_report.json",
                mime="application/json",
            )

            # -----------------------------------------
            # FINVOICE PRIORITY AGENT: AUTONOMOUS DECISION
            # -----------------------------------------
            st.divider()
            st.subheader("🎯 FinVoice Priority Agent — Autonomous Decision")
            st.caption("Real-time behavioral recovery evaluation & calling priority")

            try:
                import sys
                agent_dir = os.path.abspath("Priority Agent/priorityAgent2")
                if agent_dir not in sys.path:
                    sys.path.insert(0, agent_dir)
                from integrations.sarvam_adapter import sarvam_adapter
                from engine.monitor import monitor

                # 1. Check if multi-borrower call sheet / diary was uploaded
                borrower_list = sarvam_adapter.parse_collections_sheet(result["extracted_text"])

                if borrower_list:
                    st.markdown("### 🏆 Who Should Be Called 1st?")
                    top_b = borrower_list[0]
                    st.success(
                        f"👉 **CALL {top_b['borrower_name'].upper()} ({top_b['phone']}) FIRST (Rank #1)!**\n\n"
                        f"**Reason:** {top_b['manager_explanation']}"
                    )
                    
                    priority_table = []
                    for b in borrower_list:
                        priority_table.append({
                            "Calling Priority": f"#{b['queue_rank']}",
                            "Borrower Name": b["borrower_name"],
                            "Phone": b["phone"],
                            "Amount Overdue": f"₹{b['amount_overdue']:,.0f}",
                            "Action": b["recommended_action"],
                            "Auditable Explanation": b["manager_explanation"]
                        })
                    st.dataframe(priority_table, use_container_width=True, hide_index=True)

                else:
                    # Single borrower receipt or note
                    intent_data = sarvam_adapter.parse_speech_or_ocr_transcript(result["extracted_text"])
                    st.write(f"**Detected Intent:** `{intent_data['detected_intent']}` | **Action:** `{intent_data['recommended_action']}`")
                    if intent_data.get("extracted_amount", 0) > 0:
                        st.write(f"**Extracted Amount:** ₹{intent_data['extracted_amount']:,.0f}")

                # Optional manual dispatch to live portfolio DB
                with st.expander("Update Live Portfolio Database / Specific Loan"):
                    loan_id = st.text_input("Target Loan ID", value="LN00001", key="target_loan_input")
                    if st.button("Update Database State & Re-rank Full Queue"):
                        intent_data = sarvam_adapter.parse_speech_or_ocr_transcript(result["extracted_text"])
                        intent_data["source"] = "SARVAM_VISION_OCR"
                        intent_data["detected_languages"] = result["detected_languages"]
                        agent_res = monitor.handle_intent_signal(loan_id=loan_id, intent_data=intent_data)
                        if agent_res.get("success"):
                            st.success(f"Updated Loan {loan_id}: {agent_res['action_taken']}")
                            st.write("#### Re-ranked Portfolio Top 5:")
                            st.dataframe(agent_res.get("top_5_queue", []), use_container_width=True)

            except Exception as e:
                st.warning(f"Priority Agent evaluation note: {e}")

        except Exception as error:
            st.error(f"OCR processing failed: {error}")
            st.exception(error)

st.divider()
st.caption(
    "FinVoice OCR Lab | Sarvam Document AI | Single-Upload Multilingual Engine"
)
