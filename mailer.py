import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DEPARTMENT_EMAILS = {
    "CSE": "arun877865@gmail.com",
    "EEE": "arunkumar7904334@gmail.com",
    "MECH": "1989indhusri@gmail.com",
    "CIVIL": "adhithiee2907@gmail.com"
}

def send_summary_to_department(summary, department, document_name=None):
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    smtp_server = (os.getenv("SMTP_SERVER") or "smtp.gmail.com").strip()
    smtp_port = os.getenv("SMTP_PORT") or "587"

    if not all([sender, password, smtp_server, smtp_port]):
        raise RuntimeError("❌ Missing email environment variables (EMAIL_SENDER / EMAIL_PASSWORD)")

    receiver = DEPARTMENT_EMAILS.get(department)

    if not receiver:
        raise ValueError(f"❌ No email configured for department: {department}")

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = f"📄 New Document Routed to {department}"

    doc_label = os.path.basename(document_name) if document_name else "Processed Document"

    msg.set_content(
        f"""
Dear {department} Department,

A new document has been routed to your department.

📄 Document: {doc_label}

📝 Summary:
{summary}

Regards,
RouteX AI Routing System
"""
    )

    # ---------- ATTACH PDF IF AVAILABLE ----------
    if document_name and os.path.exists(document_name):
        try:
            with open(document_name, "rb") as f:
                pdf_data = f.read()
                pdf_name = os.path.basename(document_name)

            msg.add_attachment(
                pdf_data,
                maintype="application",
                subtype="pdf",
                filename=pdf_name
            )
            print(f"📎 Attached PDF file: {pdf_name}")
        except Exception as e:
            print(f"⚠️ Could not attach PDF file: {e}")

    # ---------- SEND EMAIL ----------
    with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)

    print(f"✅ Email sent successfully to {department} ({receiver})")
    return True
