import streamlit as st
import pandas as pd
import requests
from transformers import pipeline

# Hugging Face Sentiment Pipeline
sentiment_pipe = pipeline("text-classification", model="siebert/sentiment-roberta-large-english")

# Gemini API Connection (Replace with real API key and endpoint)
GEMINI_API_KEY = "your-gemini-api-key"
GEMINI_ENDPOINT = "https://api.your-gemini-endpoint.com/analyze"

def call_gemini_api(prompt):
    response = requests.post(
        GEMINI_ENDPOINT,
        json={"prompt": prompt},
        headers={"Authorization": f"Bearer {GEMINI_API_KEY}"}
    )
    return response.json()

# Sentiment Analysis Function
def sentiment_analysis(texts):
    results = sentiment_pipe(texts)
    positive = sum(1 for r in results if r['label'] == 'POSITIVE')
    negative = sum(1 for r in results if r['label'] == 'NEGATIVE')
    total = len(results)
    neutral = total - positive - negative
    score = round((positive - negative) / total * 10, 2) if total else 0
    return {
        "score": score,
        "positive": f"{(positive / total) * 100:.2f}%" if total else "0%",
        "neutral": f"{(neutral / total) * 100:.2f}%" if total else "0%",
        "negative": f"{(negative / total) * 100:.2f}%" if total else "0%",
    }

# Sidebar Navigation
st.sidebar.title("👨‍👩‍👧‍👦 Parent Dashboard")
page = st.sidebar.radio("Navigate", ["📊 Overview", "❤️ Wellbeing", "🧠 Smart Parenting"])

# Data Upload (Used in All Pages)
uploaded_file = st.sidebar.file_uploader("Upload Email CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Uploaded {len(df)} emails")
else:
    df = None

# Overview Page
if page == "📊 Overview":
    st.header("📊 Overview")

    if df is not None:
        st.subheader("Activity Overview")

        # Metrics Derived From Data
        total_emails = len(df)
        today_emails = df.head(3)  # Simulating "today's" emails
        screen_time = f"{total_emails * 5} min"  # Example: 5 min per email

        st.metric("📧 Total Emails", f"{total_emails} (Today: {len(today_emails)})")
        st.metric("⏰ Estimated Screen Time", screen_time)
        st.metric("🔍 Safety Score", "Calculated in Wellbeing Tab")
        st.dataframe(df)
    else:
        st.info("Please upload a CSV to view activity overview.")

# Wellbeing Page
elif page == "❤️ Wellbeing":
    st.header("❤️ Wellbeing Analysis")

    if df is not None:
        st.write("### Uploaded Emails", df)

        # Preparing Full Text for Analysis
        combined_text = df[['direction', 'sender', 'recipient', 'subject', 'body']].astype(str).agg(' '.join, axis=1).tolist()
        full_text = " ".join(combined_text)

        # Call Gemini API for Risk and Tone
        gemini_response = call_gemini_api(f"Analyze the following school emails for risk and communication tone:\n{full_text}")

        st.subheader("⚠️ Risk Assessment")
        st.write(gemini_response.get("risk_assessment", {}))

        st.subheader("🎭 Communication Tone")
        st.write(gemini_response.get("communication_tone", {}))

        # Sentiment Analysis
        sentiment_result = sentiment_analysis(combined_text)
        st.subheader("📊 Sentiment Analysis")
        st.write(sentiment_result)
    else:
        st.info("Please upload a CSV to analyze wellbeing.")

# Smart Parenting Page
elif page == "🧠 Smart Parenting":
    st.header("🧠 Smart Parenting Assistant")

    user_input = st.text_input("Ask your question about your child's digital wellbeing:")

    if st.button("Send") and user_input:
        response = call_gemini_api(user_input)
        st.write(response.get("response", "No response received."))

