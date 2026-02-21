import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from deep_translator import GoogleTranslator
import re
import uuid

# =====================================
# Page Config
# =====================================

st.set_page_config(
    page_title="Spam Detection Chat",
    page_icon="🤖",
    layout="wide"
)

# =====================================
# Language Selection (FIRST)
# =====================================

language = st.sidebar.selectbox("🌍 Language / اللغة", ["English", "العربية"])

# Translation Dictionary
translations = {
    "English": {
        "title": "🤖 Spam Detection Model",
        "conversations": "💬 Conversations",
        "new_chat": "➕ New Chat",
        "chat": "Chat",
        "placeholder": "Type your message...",
        "spam_title": "🚨 SPAM DETECTED",
        "ham_title": "✅ SAFE MESSAGE",
        "prob_spam": "Spam Probability",
        "prob_ham": "Ham Probability"
    },
    "العربية": {
        "title": "🤖 نموذج كشف الرسائل المزعجة",
        "conversations": "💬 المحادثات",
        "new_chat": "➕ محادثة جديدة",
        "chat": "محادثة",
        "placeholder": "اكتب رسالتك...",
        "spam_title": "🚨 تم اكتشاف رسالة مزعجة",
        "ham_title": "✅ رسالة آمنة",
        "prob_spam": "احتمال الإزعاج",
        "prob_ham": "احتمال السلامة"
    }
}

t = translations[language]

# =====================================
# RTL Support
# =====================================

if language == "العربية":
    st.markdown("""
        <style>
        html, body, [class*="css"] {
            direction: rtl;
            text-align: right;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        html, body, [class*="css"] {
            direction: ltr;
            text-align: left;
        }
        </style>
    """, unsafe_allow_html=True)

st.markdown(f"<h1 style='text-align:center;'>{t['title']}</h1>", unsafe_allow_html=True)
st.markdown("---")

# =====================================
# Load Dataset & Train Model
# =====================================

df = pd.read_csv("spam_cleaned.csv")
X = df["message"]
y = df["label"]

vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    max_df=0.9
)

X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

# =====================================
# Session State
# =====================================

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = []
    st.session_state.current_chat = chat_id

# =====================================
# Sidebar
# =====================================

st.sidebar.title(t["conversations"])

if st.sidebar.button(t["new_chat"]):
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = []
    st.session_state.current_chat = chat_id
    st.rerun()

st.sidebar.markdown("---")

for chat_id in st.session_state.chats:
    chat_index = list(st.session_state.chats.keys()).index(chat_id) + 1
    if st.sidebar.button(f"{t['chat']} {chat_index}", key=chat_id):
        st.session_state.current_chat = chat_id
        st.rerun()

# =====================================
# Display Messages
# =====================================

messages = st.session_state.chats[st.session_state.current_chat]

for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =====================================
# Text Cleaning
# =====================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# =====================================
# Chat Input
# =====================================

user_input = st.chat_input(t["placeholder"])

if user_input:

    messages.append({"role": "user", "content": user_input})

    # Translate Arabic input to English for model
    if language == "العربية":
        translated_input = GoogleTranslator(source='auto', target='en').translate(user_input)
    else:
        translated_input = user_input

    cleaned_input = clean_text(translated_input)
    input_vector = vectorizer.transform([cleaned_input])

    prediction = model.predict(input_vector)[0]
    probabilities = model.predict_proba(input_vector)[0]

    spam_prob = probabilities[1] * 100
    ham_prob = probabilities[0] * 100

    if prediction == 1:
        response = f"""
{t['spam_title']}

{t['prob_spam']}: {spam_prob:.2f}%
{t['prob_ham']}: {ham_prob:.2f}%
"""
    else:
        response = f"""
{t['ham_title']}

{t['prob_ham']}: {ham_prob:.2f}%
{t['prob_spam']}: {spam_prob:.2f}%
"""

    messages.append({"role": "assistant", "content": response})

    st.rerun()