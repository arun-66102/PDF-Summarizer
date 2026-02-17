import streamlit as st
import os
import sys
import time
from io import BytesIO
import tempfile
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import process_pdf, process_text

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RouteX",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# PREMIUM CSS — ChatGPT / Gemini Dark Theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Text contrast — dark on light blossom ── */
    .stApp, .stApp p, .stApp span, .stApp li, .stApp td, .stApp th,
    .stApp label, .stApp div {
        color: #2B1015 !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #3D1525 !important;
    }
    .stApp strong, .stApp b {
        color: #5C2340 !important;
    }
    .stApp a {
        color: #D4536A !important;
    }
    .stApp code {
        color: #5C2340 !important;
        background: rgba(255,255,255,0.5) !important;
    }

    /* Markdown text inside chat messages */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #2B1015 !important;
    }
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] b {
        color: #5C2340 !important;
    }

    /* Streamlit captions / small text */
    .stApp .stCaption, .stApp [data-testid="stCaptionContainer"] {
        color: #8B5A6A !important;
    }

    /* Result section text override — stays readable on mint */
    .result-section p, .result-section span, .result-section li,
    .result-section div {
        color: #1A3D20 !important;
    }
    .routing-section p, .routing-section span, .routing-section div {
        color: #1A3D20 !important;
    }
    .email-section p, .email-section span, .email-section div {
        color: #3D1525 !important;
    }

    /* Download button text */
    .stApp .stDownloadButton button {
        color: #3D1525 !important;
        border-color: #FFB7C5 !important;
        background: rgba(255,255,255,0.7) !important;
    }
    .stApp .stDownloadButton button:hover {
        background: #FFB7C5 !important;
    }

    /* Error / success / info boxes */
    .stApp .stAlert p {
        color: inherit !important;
    }
    /* ── Global Streamlit button override ── */
    .stApp button[kind="secondary"],
    .stApp button[kind="primary"],
    .stApp .stButton > button {
        background: rgba(255,255,255,0.75) !important;
        color: #5C2340 !important;
        border: 1px solid #FFB7C5 !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stApp .stButton > button:hover {
        background: #FFB7C5 !important;
        color: #3D1525 !important;
        border-color: #FF8FA3 !important;
    }

    /* ── Chat input container — remove dark wrapper ── */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background: transparent !important;
        background-color: transparent !important;
    }
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottomBlockContainer"] > div {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* ── Chat input bar ── */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] textarea {
        background: #FFFFFF !important;
        color: #2B1015 !important;
        border: 2px solid #FFB7C5 !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #B07080 !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #FF8FA3 !important;
        box-shadow: 0 0 0 3px rgba(255,183,197,0.25) !important;
    }
    [data-testid="stChatInput"] button {
        background: #FFB7C5 !important;
        color: #3D1525 !important;
        border: none !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatInput"] button:hover {
        background: #FF8FA3 !important;
    }

    /* ── Sidebar selectbox / dropdown ── */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
        background: #7A3358 !important;
        border: 1px solid #9E4570 !important;
        border-radius: 8px !important;
        color: #FFE0E8 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div:hover {
        border-color: #FFB7C5 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
        fill: #FFB7C5 !important;
    }

    /* ── Sidebar checkbox ── */
    section[data-testid="stSidebar"] [data-testid="stCheckbox"] span[data-testid="stCheckbox"]  {
        color: #FFE0E8 !important;
    }
    section[data-testid="stSidebar"] .stCheckbox > label > div[data-testid="stMarkdownContainer"] p {
        color: #FFE0E8 !important;
    }
    section[data-testid="stSidebar"] input[type="checkbox"] {
        accent-color: #FFB7C5 !important;
    }

    /* ── Sidebar slider ── */
    section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"] {
        color: #FFE0E8 !important;
    }
    section[data-testid="stSidebar"] .stSlider > div > div > div > div {
        background: #FFB7C5 !important;
    }

    /* ── File uploader / drop zone ── */
    .stApp [data-testid="stFileUploader"],
    .stApp [data-testid="stFileUploader"] div,
    .stApp [data-testid="stFileUploader"] span,
    .stApp [data-testid="stFileUploader"] p,
    .stApp [data-testid="stFileUploader"] label,
    .stApp [data-testid="stFileUploader"] small,
    .stApp [data-testid="stFileUploader"] section,
    .stApp [data-testid="stFileUploader"] section > div,
    .stApp [data-testid="stFileUploader"] section > div > span,
    .stApp [data-testid="stFileUploader"] section > div > div,
    .stApp [data-testid="stFileUploader"] section > div > small,
    .stApp [data-testid="stFileUploader"] section > button,
    .stApp [data-testid="stFileUploader"] button,
    .stApp [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
    }
    .stApp [data-testid="stFileUploader"] section {
        background: #5C2340 !important;
        border: 2px dashed #FFB7C5 !important;
        border-radius: 12px !important;
    }
    .stApp [data-testid="stFileUploader"] section:hover {
        border-color: #FFFFFF !important;
        background: #6E2D50 !important;
    }
    /* Uploaded file name chip */
    .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
    .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] div,
    .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span,
    .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] small {
        color: #FFFFFF !important;
    }
    .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button svg {
        fill: #FFB7C5 !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Cherry Blossom background ── */
    .stApp {
        background: #F2C7C7;
    }

    /* ── Sidebar — deep rose ── */
    section[data-testid="stSidebar"] {
        background: #5C2340;
        border-right: 1px solid #7A3358;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #F2C7C7 !important;
        font-size: 0.82rem !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #FFE0E8 !important;
    }
    section[data-testid="stSidebar"] a {
        color: #FFB7C5 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #7A3358;
    }
    section[data-testid="stSidebar"] button {
        color: #FFE0E8 !important;
    }

    /* ── Centered chat container ── */
    .block-container {
        max-width: 780px !important;
        margin: 0 auto;
        padding-top: 1rem !important;
        padding-bottom: 6rem !important;
    }

    /* ── Welcome hero ── */
    .welcome-hero {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 55vh;
        text-align: center;
        padding: 2rem 1rem;
    }
    .welcome-hero .logo {
        width: 64px; height: 64px;
        background: linear-gradient(135deg, #FFB7C5, #FF8FA3);
        border-radius: 18px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.8rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 30px rgba(255,183,197,0.4);
    }
    .welcome-hero .logo-img {
        width: 120px;
        height: auto;
        margin-bottom: 1.5rem;
        filter: drop-shadow(0 8px 24px rgba(255,183,197,0.35));
        transition: transform 0.3s ease;
    }
    .welcome-hero .logo-img:hover {
        transform: scale(1.05);
    }
    .welcome-hero h2 {
        color: #3D1525;
        font-weight: 700;
        font-size: 1.7rem;
        margin: 0 0 0.4rem;
        letter-spacing: -0.02em;
    }
    .welcome-hero .sub {
        color: #8B5A6A;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    .suggestion-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.6rem;
        width: 100%;
        max-width: 520px;
    }
    .suggestion-card {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.6);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        text-align: left;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .suggestion-card:hover {
        border-color: #FFB7C5;
        background: rgba(255,255,255,0.92);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255,183,197,0.2);
    }
    .suggestion-card .s-icon {
        font-size: 1.1rem;
        margin-bottom: 0.3rem;
    }
    .suggestion-card .s-title {
        color: #3D1525;
        font-size: 0.82rem;
        font-weight: 600;
        line-height: 1.35;
    }
    .suggestion-card .s-desc {
        color: #8B5A6A;
        font-size: 0.72rem;
        margin-top: 0.15rem;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.75rem 0 !important;
        max-width: 100%;
    }

    /* Chat avatar icons */
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
    [data-testid="stChatMessage"] img[data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(135deg, #FFB7C5, #FF8FA3) !important;
        border-radius: 12px !important;
        box-shadow: 0 3px 10px rgba(255,183,197,0.3);
    }
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"],
    [data-testid="stChatMessage"] img[data-testid="chatAvatarIcon-assistant"] {
        background: linear-gradient(135deg, #5C2340, #7A3358) !important;
        border-radius: 12px !important;
        box-shadow: 0 3px 10px rgba(92,35,64,0.3);
    }
    /* Avatar container */
    [data-testid="stChatMessage"] > div:first-child {
        background: transparent !important;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarContainer"] {
        background: transparent !important;
    }

    /* User message bubble — blossom pink */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: transparent !important;
    }
    .user-bubble {
        background: linear-gradient(135deg, #FFB7C5, #FF8FA3);
        border: none;
        border-radius: 18px 18px 4px 18px;
        padding: 0.75rem 1rem;
        color: #3D1525;
        font-size: 0.9rem;
        font-weight: 500;
        line-height: 1.55;
        display: inline-block;
        max-width: 85%;
        float: right;
        clear: both;
        box-shadow: 0 4px 14px rgba(255,183,197,0.3);
    }
    .user-time {
        text-align: right;
        color: #B07080;
        font-size: 0.68rem;
        margin-top: 0.25rem;
        clear: both;
    }

    /* Assistant message */
    .assistant-header {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .assistant-header .name {
        font-size: 0.78rem;
        font-weight: 700;
        color: #5C2340;
    }
    .assistant-header .time {
        font-size: 0.68rem;
        color: #B07080;
    }

    /* ── Result sections — mint green glass ── */
    .result-section {
        background: rgba(213,243,216,0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(213,243,216,0.7);
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin: 0.6rem 0;
    }
    .result-section h4 {
        margin: 0 0 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #2A5A30;
    }
    .routing-section {
        background: linear-gradient(135deg, rgba(255,255,255,0.65), rgba(213,243,216,0.4));
        border: 1px solid #A8E6B0;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin: 0.6rem 0;
    }
    .email-section {
        background: linear-gradient(135deg, rgba(255,255,255,0.65), rgba(255,183,197,0.2));
        border: 1px solid #FFB7C5;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin: 0.6rem 0;
    }

    /* ── Doc chips — white pill ── */
    .doc-chip-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 0.4rem 0;
    }
    .doc-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: #FFFFFF;
        border: 1px solid #F2C7C7;
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        font-size: 0.76rem;
        font-weight: 500;
        color: #5C2340;
        animation: fadeUp 0.3s ease;
    }
    .doc-chip .chip-size {
        color: #B07080;
        font-size: 0.66rem;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Sidebar badges ── */
    .badge-ok {
        display: inline-flex; align-items: center; gap: 0.25rem;
        background: rgba(213,243,216,0.3); border: 1px solid #A8E6B0;
        color: #4ade80; border-radius: 8px;
        padding: 0.2rem 0.55rem; font-size: 0.72rem; font-weight: 500;
    }
    .badge-warn {
        display: inline-flex; align-items: center; gap: 0.25rem;
        background: rgba(255,183,100,0.2); border: 1px solid #FFB76450;
        color: #fb923c; border-radius: 8px;
        padding: 0.2rem 0.55rem; font-size: 0.72rem; font-weight: 500;
    }
    .badge-err {
        display: inline-flex; align-items: center; gap: 0.25rem;
        background: rgba(255,100,100,0.2); border: 1px solid #FF646450;
        color: #f87171; border-radius: 8px;
        padding: 0.2rem 0.55rem; font-size: 0.72rem; font-weight: 500;
    }

    /* ── New Chat button — Cherry Blossom Theme ── */
    .new-chat-btn button {
        background: #7A3358 !important;
        border: 1px solid #FFB7C5 !important;
        color: #FFE0E8 !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.4rem 1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(92,35,64,0.3) !important;
    }
    .new-chat-btn button:hover {
        background: #FFB7C5 !important;
        color: #3D1525 !important;
        border-color: #FF8FA3 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(255,183,197,0.3) !important;
    }

    /* ── Sidebar selectbox / dropdown — refined ── */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
        background: #5C2340 !important;
        border: 1px solid #FFB7C5 !important;
        border-radius: 10px !important;
        color: #FFE0E8 !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div:hover {
        border-color: #FFFFFF !important;
        background: #6E2D50 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
        fill: #FFB7C5 !important;
    }
    /* Dropdown menu items */
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background: #5C2340 !important;
        border: 1px solid #FFB7C5 !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li {
        background: #5C2340 !important;
        color: #FFE0E8 !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover,
    ul[data-testid="stSelectboxVirtualDropdown"] li[aria-selected="true"] {
        background: #FFB7C5 !important;
        color: #3D1525 !important;
    }

    /* ── Metrics styling — white glass ── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.5);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
    }
    [data-testid="stMetricLabel"] {
        color: #8B5A6A !important;
        font-size: 0.72rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #3D1525 !important;
        font-size: 0.95rem !important;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 0.5rem 0 1rem;
        color: #B07080;
        font-size: 0.7rem;
    }
    .app-footer span { color: #D4536A; font-weight: 600; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: #FFB7C5;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #FF8FA3; }

    /* ── Message fade-in ── */
    @keyframes msgFade {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stChatMessage"] {
        animation: msgFade 0.35s ease;
    }

    /* ── Expander in sidebar ── */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        color: #F2C7C7 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }

    /* ── Doc send button — blossom gradient ── */
    .doc-send-btn button {
        background: linear-gradient(135deg, #FFB7C5, #FF8FA3) !important;
        color: #3D1525 !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        padding: 0.55rem 1.25rem !important;
        animation: pulse-send 1.8s ease-in-out infinite;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 16px rgba(255,183,197,0.35);
    }
    .doc-send-btn button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 6px 28px rgba(255,143,163,0.4) !important;
        background: linear-gradient(135deg, #FF8FA3, #FF6B85) !important;
    }
    @keyframes pulse-send {
        0%, 100% { box-shadow: 0 4px 16px rgba(255,183,197,0.3); }
        50% { box-shadow: 0 4px 24px rgba(255,183,197,0.5); }
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "attached_docs" not in st.session_state:
    st.session_state.attached_docs = []


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Minimal Settings Panel
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── New Chat button ──
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("✦  New Chat", key="new_chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.attached_docs = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Model selector ──
    model_options = {
        "llama3-8b": "Llama 3.1 8B · Fast",
        "llama3-70b": "Llama 3.3 70B · Quality",
        "llama-guard": "Llama Guard 4 · Filter",
        "gpt-oss-20b": "GPT-OSS 20B",
        "gpt-oss-120b": "GPT-OSS 120B"
    }
    selected_model = st.selectbox(
        "Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0,
        label_visibility="collapsed"
    )

    # ── Settings expander ──
    with st.expander("⚙ Settings", expanded=False):
        context_limit = st.slider(
            "Context limit", 1000, 8000, 4000, 500,
            help="Max context window for the AI model"
        )
        enable_routing = st.checkbox("Department routing", value=True)
        enable_email = st.checkbox("Email delivery", value=True)

    st.markdown("---")

    # ── API status (subtle) ──
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        st.markdown('<span class="badge-ok">● Groq connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-err">● Groq key missing</span>', unsafe_allow_html=True)

    if enable_email:
        email_sender = os.getenv("EMAIL_SENDER")
        if email_sender:
            st.markdown('<span class="badge-ok">● Email ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-warn">● Email not set</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Helper to load image as base64 ──
import base64
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

if not st.session_state.messages:
    # ── Welcome hero ──
    try:
        logo_base64 = get_img_as_base64("logo.png")
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="logo-img">'
    except Exception:
        logo_html = '<div class="logo">🤖</div>'

    st.markdown(f"""
    <div class="welcome-hero">
        {logo_html}
        <h2>What can I help with?</h2>
        <div class="sub">Upload a PDF or paste text — I'll summarize, analyze, and route it.</div>
        <div class="suggestion-grid">
            <div class="suggestion-card">
                <div class="s-icon">📄</div>
                <div class="s-title">Summarize PDF</div>
                <div class="s-desc">Get a quick overview of any document</div>
            </div>
            <div class="suggestion-card">
                <div class="s-icon">🔍</div>
                <div class="s-title">Analyze text</div>
                <div class="s-desc">Extract key insights and action items</div>
            </div>
            <div class="suggestion-card">
                <div class="s-icon">🎯</div>
                <div class="s-title">Route to department</div>
                <div class="s-desc">Classify content for specific teams</div>
            </div>
            <div class="suggestion-card">
                <div class="s-icon">📧</div>
                <div class="s-title">Email summary</div>
                <div class="s-desc">Send the results directly to your inbox</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # ── Render chat history ──
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                # Doc chip
                if msg.get("doc_name"):
                    st.markdown(f'<div class="doc-chip-bar"><div class="doc-chip">📄 {msg["doc_name"]}</div></div>', unsafe_allow_html=True)
                # Text bubble
                if msg.get("text"):
                    preview = msg["text"][:500]
                    if len(msg["text"]) > 500:
                        preview += "…"
                    st.markdown(f'<div class="user-bubble">{preview}</div><div class="user-time">{msg.get("time", "")}</div>', unsafe_allow_html=True)
                elif msg.get("doc_name"):
                    st.markdown(f'<div class="user-time">{msg.get("time", "")}</div>', unsafe_allow_html=True)

        else:
            with st.chat_message("assistant", avatar="🤖"):
                result = msg.get("result", {})

                st.markdown(f'<div class="assistant-header"><span class="name">RouteX</span><span class="time">{msg.get("time", "")}</span></div>', unsafe_allow_html=True)

                if "error" in result and result.get("routing") is None:
                    st.error(f"❌ {result.get('summary', result.get('error', 'Unknown error'))}")
                else:
                    # Summary
                    st.markdown('<div class="result-section">', unsafe_allow_html=True)
                    st.markdown(f"**📝 Summary**\n\n{result.get('summary', 'No summary generated')}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Routing
                    routing = result.get("routing", {})
                    primary_depts = routing.get("primary_departments", []) if routing else []
                    if primary_depts:
                        dept_str = ", ".join(primary_depts)
                        tie_label = " (tie)" if routing.get("is_tie") else ""
                        st.markdown(f'<div class="routing-section">', unsafe_allow_html=True)
                        st.markdown(f"**🎯 Routed to{tie_label}:** {dept_str}")
                        st.caption(f"{routing.get('method', '?')} · confidence {routing.get('confidence', 0):.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Metrics
                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        st.metric("Length", f"{result.get('text_length', 0)}")
                    with mc2:
                        st.metric("Chunks", result.get("chunks_processed", 0))
                    with mc3:
                        st.metric("Model", result.get("model_used", "?"))

                    # Download
                    summary_text = result.get("summary", "")
                    if summary_text:
                        st.download_button(
                            "💾 Download",
                            data=summary_text,
                            file_name=f"summary_{msg.get('time', '').replace(':', '').replace(' ', '_')}.txt",
                            mime="text/plain",
                            key=f"dl_{msg.get('time', '')}"
                        )


# ══════════════════════════════════════════════════════════════════════════════
# ATTACH BAR (above chat_input)
# ══════════════════════════════════════════════════════════════════════════════

# Show attached doc chips
if st.session_state.attached_docs:
    chips_html = '<div class="doc-chip-bar">'
    for doc in st.session_state.attached_docs:
        size_kb = doc["size"] / 1024
        chips_html += f'<div class="doc-chip">📄 {doc["name"]} <span class="chip-size">({size_kb:.1f} KB)</span></div>'
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)

    # Remove buttons
    rm_cols = st.columns(len(st.session_state.attached_docs) + 1)
    for i, doc in enumerate(st.session_state.attached_docs):
        with rm_cols[i]:
            if st.button(f"✕ {doc['name'][:12]}", key=f"rm_doc_{i}", help=f"Remove {doc['name']}"):
                st.session_state.attached_docs.pop(i)
                st.rerun()
    with rm_cols[-1]:
        if st.button("Clear all", key="clear_all_docs"):
            st.session_state.attached_docs = []
            st.rerun()

# 📎 File uploader (compact, just above the chat input)
uploaded_file = st.file_uploader(
    "📎 Attach PDF",
    type=["pdf"],
    help="Attach a PDF document to summarize",
    key="inline_pdf_uploader",
    label_visibility="collapsed"
)
if uploaded_file is not None:
    existing = [d["name"] for d in st.session_state.attached_docs]
    if uploaded_file.name not in existing:
        st.session_state.attached_docs.append({
            "name": uploaded_file.name,
            "size": uploaded_file.size,
            "data": uploaded_file.getvalue()
        })
        st.rerun()

# ── Send button when docs are attached ──
doc_send_clicked = False
if st.session_state.attached_docs:
    st.markdown('<div class="doc-send-btn">', unsafe_allow_html=True)
    doc_send_clicked = st.button(
        f"➤  Process {len(st.session_state.attached_docs)} document(s)",
        key="doc_send_btn",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT (native Streamlit — enter to send, pinned to bottom)
# ══════════════════════════════════════════════════════════════════════════════
prompt = st.chat_input("Message RouteX…")

if prompt or doc_send_clicked:
    now_str = datetime.now().strftime("%I:%M %p")
    has_docs = len(st.session_state.attached_docs) > 0
    has_text = bool(prompt and prompt.strip())

    # ── Decide mode ──
    if has_docs:
        process_pdf_mode = True
        doc_data = st.session_state.attached_docs[0]
        doc_name = doc_data["name"]
    else:
        process_pdf_mode = False
        doc_name = None

    # ── Add user message to history ──
    st.session_state.messages.append({
        "role": "user",
        "text": prompt if has_text else None,
        "doc_name": doc_name,
        "time": now_str
    })

    # ── Show user message ──
    with st.chat_message("user", avatar="🧑‍💻"):
        if doc_name:
            st.markdown(f'<div class="doc-chip-bar"><div class="doc-chip">📄 {doc_name}</div></div>', unsafe_allow_html=True)
        if has_text:
            preview = prompt[:500] + ("…" if len(prompt) > 500 else "")
            st.markdown(f'<div class="user-bubble">{preview}</div><div class="user-time">{now_str}</div>', unsafe_allow_html=True)

    # ── Process and show assistant response ──
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(f'<div class="assistant-header"><span class="name">RouteX</span><span class="time">{now_str}</span></div>', unsafe_allow_html=True)

        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        tmp_file_path = None

        try:
            if process_pdf_mode:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(doc_data["data"])
                    tmp_file_path = tmp_file.name

                progress_bar.progress(10)
                status_placeholder.caption("📄 Extracting text from PDF...")

                def progress_callback(current, total):
                    pct = min(0.9, 0.1 + (current / total) * 0.7)
                    progress_bar.progress(pct)
                    status_placeholder.caption(f"🔄 Processing chunk {current}/{total}...")

                progress_bar.progress(20)
                status_placeholder.caption("🧠 Generating summary...")

                result = process_pdf(
                    tmp_file_path,
                    model=selected_model,
                    model_context_limit=context_limit,
                    progress_callback=progress_callback,
                    enable_routing=enable_routing
                )
            else:
                progress_bar.progress(10)
                status_placeholder.caption("✏️ Preparing text...")

                def progress_callback(current, total):
                    pct = min(0.9, 0.1 + (current / total) * 0.7)
                    progress_bar.progress(pct)
                    status_placeholder.caption(f"🔄 Processing chunk {current}/{total}...")

                progress_bar.progress(20)
                status_placeholder.caption("🧠 Generating summary...")

                result = process_text(
                    prompt,
                    model=selected_model,
                    model_context_limit=context_limit,
                    progress_callback=progress_callback,
                    enable_routing=enable_routing
                )

            progress_bar.progress(100)
            status_placeholder.caption("✅ Done")
            time.sleep(0.4)
            status_placeholder.empty()
            progress_bar.empty()

            # ── Display results ──
            if "error" in result and result.get("routing") is None:
                st.error(f"❌ {result.get('summary', result.get('error', 'Unknown error'))}")
            else:
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                st.markdown(f"**📝 Summary**\n\n{result.get('summary', 'No summary generated')}")
                st.markdown('</div>', unsafe_allow_html=True)

                routing = result.get("routing", {})
                primary_depts = routing.get("primary_departments", []) if routing else []
                if primary_depts:
                    dept_str = ", ".join(primary_depts)
                    tie_label = " (tie)" if routing.get("is_tie") else ""
                    st.markdown('<div class="routing-section">', unsafe_allow_html=True)
                    st.markdown(f"**🎯 Routed to{tie_label}:** {dept_str}")
                    st.caption(f"{routing.get('method', '?')} · confidence {routing.get('confidence', 0):.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Email
                    if enable_email and process_pdf_mode and tmp_file_path:
                        st.markdown('<div class="email-section">', unsafe_allow_html=True)
                        from main import send_pdf_to_departments
                        email_sent = send_pdf_to_departments(
                            tmp_file_path, result["summary"], routing
                        )
                        if email_sent:
                            st.success("📧 Emails sent")
                        else:
                            st.warning("⚠️ Some emails failed")
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif enable_email and not process_pdf_mode:
                        st.caption("ℹ️ Attach a PDF for email delivery.")

                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("Length", f"{result.get('text_length', 0)}")
                with mc2:
                    st.metric("Chunks", result.get("chunks_processed", 0))
                with mc3:
                    st.metric("Model", result.get("model_used", "?"))

                summary_text = result.get("summary", "")
                if summary_text:
                    st.download_button(
                        "💾 Download",
                        data=summary_text,
                        file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key=f"dl_latest_{datetime.now().strftime('%H%M%S')}"
                    )

            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "result": result,
                "source": "pdf" if process_pdf_mode else "text",
                "time": now_str
            })

            # Clear attached docs
            st.session_state.attached_docs = []

        except Exception as e:
            progress_bar.empty()
            status_placeholder.empty()
            st.error(f"❌ Processing failed: {str(e)}")
            if tmp_file_path:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass

# ── Footer ──
st.markdown("""
<div class="app-footer">
    Powered by <span>Groq AI</span> · Intelligent Routing · Built by <span>RAG Retrievers</span>
</div>
""", unsafe_allow_html=True)