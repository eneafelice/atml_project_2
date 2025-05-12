import streamlit as st
import pandas as pd
import requests

# ---------------- Hugging Face Inference API Setup ----------------
HF_API_KEY = st.secrets.get("HF_API_KEY", "your-hf-api-key")
HF_API_URL = "https://api-inference.huggingface.co/models/siebert/sentiment-roberta-large-english"

def hf_sentiment_analysis(text):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": text}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    return response.json()

# ---------------- Gemini 1.5 Flash API via AI Studio ----------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "your-gemini-api-key")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

def call_gemini_api(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(GEMINI_URL, headers=headers, json=payload)
    try:
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ---------------- Streamlit UI ----------------
st.sidebar.title("👨‍👩‍👧‍👦 Parent Dashboard")
page = st.sidebar.radio("Navigate", ["📊 Overview", "❤️ Wellbeing", "🧠 Smart Parenting"])
uploaded_file = st.sidebar.file_uploader("Upload Email CSV", type="csv")

df = pd.read_csv(uploaded_file) if uploaded_file else None
if uploaded_file:
    st.sidebar.success(f"Uploaded {len(df)} emails")

# ---------------- Overview Page ----------------
if page == "📊 Overview":
    st.header("📊 Overview")
    if df is not None:
        st.subheader("Activity Overview")
        st.metric("📧 Total Emails", len(df))
        st.metric("⏰ Estimated Screen Time", f"{len(df) * 5} min")
        st.metric("🔍 Safety Score", "Calculated in Wellbeing Tab")
        st.dataframe(df)
    else:
        st.info("Please upload a CSV to view activity overview.")

# ---------------- Wellbeing Page ----------------
elif page == "❤️ Wellbeing":
    st.header("❤️ Wellbeing Analysis")
    if df is not None:
        st.write("### Uploaded Emails", df)
        combined_text_list = df[['direction', 'sender', 'recipient', 'subject', 'body']].astype(str).agg(' '.join, axis=1).tolist()
        full_text = " ".join(combined_text_list)

        # Gemini


