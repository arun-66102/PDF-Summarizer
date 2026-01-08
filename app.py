import streamlit as st
import os
import sys
import time
from io import BytesIO
import tempfile
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import process_pdf

# Configure Streamlit page
st.set_page_config(
    page_title="PDF Document Processor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e1e5e9;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .routing-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .email-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI-Powered Document Processor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = {
            "llama3-8b": "Llama 3.1 8B (Fastest)",
            "llama3-70b": "Llama 3.3 70B (Higher Quality)",
            "llama-guard": "Llama Guard 4 12B (Content Filter)",
            "gpt-oss-20b": "GPT-OSS 20B",
            "gpt-oss-120b": "GPT-OSS 120B"
        }
        
        selected_model = st.selectbox(
            "🤖 AI Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        
        # Context limit
        context_limit = st.slider(
            "📝 Context Limit",
            min_value=1000,
            max_value=8000,
            value=4000,
            step=500,
            help="Maximum context window for the AI model"
        )
        
        # Routing and email options
        st.subheader("📧 Routing & Email")
        enable_routing = st.checkbox("🎯 Enable Department Routing", value=True)
        enable_email = st.checkbox("📧 Send Email to Departments", value=True)
        
        # Department info
        st.subheader("🏢 Departments")
        st.markdown("""
        - **CSE**: Computer Science & Engineering
        - **EEE**: Electrical & Electronics Engineering  
        - **MECH**: Mechanical Engineering
        - **CIVIL**: Civil Engineering
        """)
        
        # API Key status
        st.subheader("🔑 API Status")
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            st.success("✅ Groq API Key configured")
        else:
            st.error("❌ Groq API Key not found")
            st.info("Set GROQ_API_KEY environment variable")
        
        # Email status
        email_sender = os.getenv("EMAIL_SENDER")
        if enable_email:
            if email_sender:
                st.success("✅ Email configuration found")
            else:
                st.warning("⚠️ Email configuration missing")
                st.info("Set EMAIL_SENDER, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.header("📄 Upload PDF Document")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document for processing"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📊 File size: {uploaded_file.size / 1024:.1f} KB")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.header("📋 Quick Info")
        st.markdown("""
        **Features:**
        - 🧠 AI-powered summarization
        - 🎯 Automatic department routing
        - 📧 Email delivery system
        - 🔄 Tie handling
        - 📊 Progress tracking
        """)
    
    # Processing section
    if uploaded_file is not None:
        st.markdown("---")
        st.header("🚀 Processing")
        
        # Process button
        if st.button("🎯 Process Document", type="primary", use_container_width=True):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Progress updates
                progress_bar.progress(10)
                status_text.text("📄 Extracting text from PDF...")
                
                def progress_callback(current, total):
                    progress = min(0.9, 0.1 + (current / total) * 0.7)
                    progress_bar.progress(progress)
                    status_text.text(f"🔄 Processing chunk {current}/{total}...")
                
                progress_bar.progress(20)
                status_text.text("🧠 Generating AI summary...")
                
                # Process the PDF
                result = process_pdf(
                    tmp_file_path,
                    model=selected_model,
                    model_context_limit=context_limit,
                    progress_callback=progress_callback,
                    enable_routing=enable_routing
                )
                
                progress_bar.progress(95)
                status_text.text("📧 Sending emails (if enabled)...")
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                
                # Display results
                st.markdown("---")
                st.header("📊 Results")
                
                # Summary section
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                st.subheader("📝 Document Summary")
                summary = result.get('summary', 'No summary generated')
                st.text_area("", value=summary, height=200, disabled=True, label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Routing information
                if enable_routing and result.get('routing'):
                    routing = result['routing']
                    primary_depts = routing.get('primary_departments', [])
                    
                    if primary_depts:
                        st.markdown('<div class="routing-box">', unsafe_allow_html=True)
                        st.subheader("🎯 Department Routing")
                        
                        if len(primary_depts) == 1:
                            st.success(f"📍 Routed to: **{primary_depts[0]}**")
                        else:
                            st.info(f"📍 Routed to multiple departments: **{', '.join(primary_depts)}**")
                        
                        # Show routing details
                        with st.expander("🔍 Routing Details"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Method", routing.get('method', 'unknown'))
                                st.metric("Confidence", f"{routing.get('confidence', 0):.3f}")
                            with col2:
                                st.metric("Is Tie", "Yes" if routing.get('is_tie') else "No")
                                st.metric("Tie Threshold", routing.get('tie_threshold', 0.05))
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Email sending
                        if enable_email and primary_depts:
                            st.markdown('<div class="email-box">', unsafe_allow_html=True)
                            st.subheader("📧 Email Delivery")
                            
                            # Send emails
                            from main import send_pdf_to_departments
                            email_sent = send_pdf_to_departments(
                                uploaded_file.name, 
                                result['summary'], 
                                routing
                            )
                            
                            if email_sent:
                                st.success("✅ Emails sent successfully to all departments!")
                            else:
                                st.warning("⚠️ Some emails may have failed")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                
                # File information
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                st.subheader("📊 Processing Details")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Text Length", f"{result.get('text_length', 0)} chars")
                with col2:
                    st.metric("Chunks", result.get('chunks_processed', 0))
                with col3:
                    st.metric("Model", selected_model)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download summary
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                st.subheader("💾 Download Summary")
                
                summary_text = result.get('summary', '')
                if summary_text:
                    st.download_button(
                        label="📄 Download Summary as Text",
                        data=summary_text,
                        file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.error(f"❌ Processing failed: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Clean up on error
                if 'tmp_file_path' in locals():
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🤖 Powered by Groq AI • 🎯 Intelligent Routing • 📧 Automated Delivery</p>
        <p>Built by RAG Retrievers</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
