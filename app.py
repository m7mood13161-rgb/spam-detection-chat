import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from deep_translator import GoogleTranslator
import re
import uuid

# ===============================
# Page Config
# ===============================

st.set_page_config(
    page_title="Spam Detection Chat",
    page_icon="🤖",
    layout="wide"
)

# ===============================
# Load Dataset & Train Model
# ===============================

@st.cache_resource
def load_model():
    df = pd.read_csv("spam_cleaned.csv")
    X = df["message"]
    y = df["label"]

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.9
    )

    X_vectorized = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vectorized, y)

    return model, vectorizer

model, vectorizer = load_model()

# ===============================
# Session State
# ===============================

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = []
    st.session_state.current_chat = chat_id

# ===============================
# Sidebar
# ===============================

st.sidebar.title("💬 Conversations")

language = st.sidebar.selectbox("🌍 Language", ["English", "العربية"])

# Proper RTL Fix (Mobile Safe)
if language == "العربية":
    st.markdown("""
        <style>
        .main {
            direction: rtl;
        }
        .block-container {
            direction: rtl;
            text-align: right;
        }
        textarea, input {
            direction: rtl !important;
            text-align: right !important;
        }
        </style>
    """, unsafe_allow_html=True)
    title_text = "🤖 نموذج كشف الرسائل المزعجة"
    input_placeholder = "اكتب رسالتك..."
else:
    title_text = "🤖 Spam Detection Model"
    input_placeholder = "Type your message..."

st.markdown(f"<h1 style='text-align:center;'>{title_text}</h1>", unsafe_allow_html=True)
st.markdown("---")

# New Chat
if st.sidebar.button("➕ New Chat" if language == "English" else "➕ محادثة جديدة"):
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = []
    st.session_state.current_chat = chat_id
    st.rerun()

st.sidebar.markdown("---")

# Chat List
for chat_id in st.session_state.chats:
    label = f"🗂 Chat {list(st.session_state.chats.keys()).index(chat_id)+1}"
    if language == "العربية":
        label = f"🗂 محادثة {list(st.session_state.chats.keys()).index(chat_id)+1}"

    if st.sidebar.button(label, key=chat_id):
        st.session_state.current_chat = chat_id
        st.rerun()

# ===============================
# Display Messages
# ===============================

messages = st.session_state.chats[st.session_state.current_chat]

for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ===============================
# Text Cleaning
# ===============================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# ===============================
# Chat Input
# ===============================

user_input = st.chat_input(input_placeholder)

if user_input:

    messages.append({"role": "user", "content": user_input})

    # Translate Arabic to English for model
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

    # Generate Response
    if prediction == 1:
        if language == "العربية":
            response = f"""
🚨 **تم اكتشاف رسالة مزعجة**

احتمال السبام: {spam_prob:.2f}%
احتمال الرسالة الآمنة: {ham_prob:.2f}%
"""
        else:
            response = f"""
🚨 **SPAM DETECTED**

Spam Probability: {spam_prob:.2f}%
Ham Probability: {ham_prob:.2f}%
"""
    else:
        if language == "العربية":
            response = f"""
✅ **رسالة آمنة**

احتمال الرسالة الآمنة: {ham_prob:.2f}%
احتمال السبام: {spam_prob:.2f}%
"""
        else:
            response = f"""
✅ **SAFE MESSAGE**

Ham Probability: {ham_prob:.2f}%
Spam Probability: {spam_prob:.2f}%
"""

    messages.append({"role": "assistant", "content": response})
    st.rerun()
