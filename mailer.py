import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv()) 
DEPARTMENT_EMAILS = {
    "CSE": "arun877865@gmail.com",
    "EEE": "arunkumar7904334@gmail.com",
    "MECH": "1989indhusri@gmail.com",
    "CIVIL": "dhan0529Ree@gmail.com"
}

def send_summary_to_department(summary, department, document_name):
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER").strip()
    smtp_port = os.getenv("SMTP_PORT")

    if not all([sender, password, smtp_server, smtp_port]):
        raise RuntimeError("❌ Missing email environment variables")

    receiver = DEPARTMENT_EMAILS.get(department)

    if not receiver:
        raise ValueError(f"❌ No email configured for department: {department}")

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = f"📄 New Document Routed to {department}"

    msg.set_content(
        f"""
Dear {department} Department,

A new document has been routed to your department.

📄 Document Name: {os.path.basename(document_name)}

📝 Summary:
{summary}

Regards,
PDF Routing System
"""
    )

    # ---------- ATTACH PDF ----------
    with open(document_name, "rb") as f:
        pdf_data = f.read()
        pdf_name = os.path.basename(document_name)

    msg.add_attachment(
        pdf_data,
        maintype="application",
        subtype="pdf",
        filename=pdf_name
    )

    # ---------- SEND EMAIL ----------
    with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)

    print("✅ Email with PDF attachment sent successfully")
