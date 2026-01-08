import os
import smtplib
from email.message import EmailMessage

DEPARTMENT_EMAILS = {
    "CSE": "1989indhusri@gmail.com",
    "EEE": "arunkumar7904334@gmail.com",
    "MECH": "1989indhusri@gmail.com",
    "CIVIL": "1989indhusri@gmail.com"
}

def send_summary_to_department(summary, department, document_name):
    try:
        # Check environment variables
        sender = os.getenv("EMAIL_SENDER")
        password = os.getenv("EMAIL_PASSWORD")
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = os.getenv("SMTP_PORT")

        print(f"🔧 Email config check - Sender: {sender}, Server: {smtp_server}, Port: {smtp_port}")

        if not all([sender, password, smtp_server, smtp_port]):
            missing = []
            if not sender: missing.append("EMAIL_SENDER")
            if not password: missing.append("EMAIL_PASSWORD")
            if not smtp_server: missing.append("SMTP_SERVER")
            if not smtp_port: missing.append("SMTP_PORT")
            raise RuntimeError(f"❌ Missing email environment variables: {', '.join(missing)}")

        receiver = DEPARTMENT_EMAILS.get(department)

        if not receiver:
            raise ValueError(f"❌ No email configured for department: {department}")

        print(f"📧 Sending email to {department} ({receiver})")

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
        print(f"📎 Attaching PDF: {document_name}")
        
        # Check if file exists
        if not os.path.exists(document_name):
            raise FileNotFoundError(f"❌ PDF file not found: {document_name}")
        
        file_size = os.path.getsize(document_name)
        print(f"📊 PDF file size: {file_size / 1024:.1f} KB")
        
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
        print(f"🔌 Connecting to SMTP server: {smtp_server}:{smtp_port}")
        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.starttls()
            print("🔐 Logging in to SMTP server...")
            server.login(sender, password)
            print("📤 Sending email...")
            server.send_message(msg)

        print(f"✅ Email sent successfully to {department} ({receiver})")
        return True

    except Exception as e:
        print(f"❌ Email sending failed for {department}: {str(e)}")
        return False
