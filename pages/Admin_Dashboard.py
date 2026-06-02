import streamlit as st

from analytics_db import init_analytics_db
from feedback_db import init_feedback_db
from watchlist_db import init_db
from stats_ui import render_statistics_dashboard

st.set_page_config(page_title="Admin Analytics", layout="wide", page_icon="📊")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #0a0c10 0%, #0E1117 50%, #12151c 100%); }
    header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; min-height: 0 !important; }
    div[data-testid="stToolbar"] { display: none !important; }
    .block-container { padding-top: 1.5rem !important; max-width: 95% !important; }
    h1, h2, h3, h4 { color: #FAFAFA !important; }
    [data-testid="stMetric"] {
        background: rgba(38, 39, 48, 0.8);
        border: 1px solid #3d4450;
        border-radius: 12px;
        padding: 1rem;
    }
    [data-testid="stMetricLabel"] { color: #9ca3af !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

init_db()
init_feedback_db()
init_analytics_db()

st.title("📊 Admin Analytics Dashboard")
st.caption("Full analytics view — same charts as the Statistics tab on the home page")

render_statistics_dashboard()
