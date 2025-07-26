import streamlit as st
import joblib
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- 1. Load the saved model and vectorizer ---
try:
    model = joblib.load('models/sentiment_model.joblib')
    tfidf = joblib.load('models/tfidf_vectorizer.joblib')
except FileNotFoundError:
    st.error("Model files not found. Please run `train_model.py` first.")
    st.stop()


# --- 2. Re-create the preprocessing function ---
# This must be the EXACT same function used during training
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)


# --- 3. Create the Streamlit App UI ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬", layout="centered")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write(
    "Enter a movie review below, and the model will predict "
    "whether the sentiment is **Positive** or **Negative**."
)

# Text area for user input
user_input = st.text_area("Enter your review here:", height=150)

# Analyze button
if st.button("Analyze Sentiment"):
    if user_input:
        # 1. Preprocess the user's input
        cleaned_input = preprocess_text(user_input)

        # 2. Vectorize the cleaned input
        input_vector = tfidf.transform([cleaned_input])

        # 3. Make a prediction
        prediction = model.predict(input_vector)[0]
        prediction_proba = model.predict_proba(input_vector)

        # 4. Display the result
        st.subheader("Analysis Result")
        if prediction == 'positive':
            st.success(f"Positive Sentiment (Confidence: {prediction_proba[0][1]:.2f})")
        else:
            st.error(f"Negative Sentiment (Confidence: {prediction_proba[0][0]:.2f})")
    else:
        st.warning("Please enter a review to analyze.")

st.markdown("---")
st.write("Built by - Jaden Isaac")
