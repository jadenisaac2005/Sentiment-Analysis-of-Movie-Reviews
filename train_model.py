import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib 

print("--- Training script started ---")

# --- 1. Load Data ---
df = pd.read_csv('data/IMDB Dataset.csv')

# --- 2. Preprocessing ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

print("Preprocessing data... (This will take a few minutes)")
df['cleaned_review'] = df['review'].apply(preprocess_text)
print("Preprocessing complete.")


# --- 3. Feature Extraction & Label Encoding ---
print("Vectorizing text...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_review'])
y = df['sentiment']
print("Vectorization complete.")


# --- 4. Model Training ---
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
print("Model training complete.")


# --- 5. Save the Model and Vectorizer ---
print("Saving model and vectorizer to 'models/' directory...")
joblib.dump(model, 'models/sentiment_model.joblib')
joblib.dump(tfidf, 'models/tfidf_vectorizer.joblib')
print("--- Training script finished successfully ---")
