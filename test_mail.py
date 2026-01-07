from mailer import send_summary_to_department

send_summary_to_department(
    summary="This is a test email with PDF attachment.",
    department="CSE",
    document_name=r"C:\Users\indhu\Downloads\STL_Delete_Functions_Explanation.pdf"
)
