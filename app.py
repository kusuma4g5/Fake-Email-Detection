import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = pickle.load(open('model/email_model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

# Preprocessing function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# Streamlit UI
st.title("📧 Fake Email Detection using Machine Learning")
st.markdown("Detect whether an email is **Spam** or **Legitimate** using NLP and ML.")

email_input = st.text_area("Paste the email content here...")

if st.button("Analyze"):
    if email_input.strip() != "":
        cleaned = clean_text(email_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        if prediction == "spam":
            st.error("⚠️ This email is likely **Fake / Spam!**")
        else:
            st.success("✅ This email seems **Legitimate!**")
    else:
        st.warning("Please enter some email text to analyze.")
