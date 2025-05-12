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
    try:
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ---------------- Gemini 1.5 Flash API Setup ----------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "your-gemini-api-key")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

def call_gemini_api(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(GEMINI_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return f"Error {response.status_code}: {response.text}"

    result = response.json()
    return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response generated.")

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

        # Gemini Analysis
        st.subheader("⚠️ Risk Assessment and Communication Tone")
        gemini_result = call_gemini_api(f"You are a content safety assistant for a school-parent dashboard. Given the text of an email written by a student, return a list of behavioral or risk-related categories that apply to the email content.Only detect categories from a predefined list. If none apply, return an empty list. Do not rewrite or summarize the email. Do not generate advice.Allowed Categories:Violence,Self-harm, Gambling, Sexual content, Inappropriate language, Scam / Phishing, Academic cheating, External contact, Late-night use (only if time is provided):\n{full_text}")
        st.write(gemini_result)

        # Hugging Face Sentiment Analysis (Sampled for Performance)
        st.subheader("📊 Sentiment Analysis (Sampled 5 Emails)")
        sentiment_results = []
        for text in combined_text_list[:5]:  # limit to first 5 for performance
            sentiment = hf_sentiment_analysis(text)
            sentiment_results.append(sentiment)
        st.write(sentiment_results)
    else:
        st.info("Please upload a CSV to analyze wellbeing.")
        
# ---------------- Smart Parenting Page ----------------
elif page == "🧠 Smart Parenting":
    st.header("🧠 Smart Parenting Assistant")

    # Auto-generate 3 suggestions on page load
    st.subheader("📝 Suggested Tips for Parents")
    auto_tips_prompt = "Provide 3 short, actionable parenting or educational tips based on the emails in the CSV File."
    auto_tips = call_gemini_api(auto_tips_prompt)
    st.write(auto_tips)

    # User input for additional questions
    st.subheader("💬 Ask for More Guidance")
    user_input = st.text_input("Ask your question about your child's wellbeing or digital behavior:")
    if st.button("Send") and user_input:
        reply = call_gemini_api(user_input)
        st.write(reply)

