import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


## ----------------------------------------------------------------
## Phase 2: Data Loading & Exploration
## ----------------------------------------------------------------
print("Phase 2: Loading Data...")
# Load the dataset from the 'data' folder
try:
    df = pd.read_csv('data/IMDB Dataset.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'data/IMDB Dataset.csv' not found. Make sure the file is in the correct directory.")
    exit() # Exit the script if the file isn't found

# Display basic info about the dataset
print("\n--- Data Exploration ---")
print(df.head()) # <-- Add this line back
df.info()
print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())
print("-" * 25)


## ----------------------------------------------------------------
## Phase 3: Data Preprocessing
## ----------------------------------------------------------------
print("\nPhase 3: Preprocessing Data...")

# Initialize the lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """A function to clean raw text data."""
    # 1. Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # 2. Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # 3. Convert to lowercase
    text = text.lower()
    # 4. Tokenize
    tokens = word_tokenize(text)
    # 5. Remove stopwords and 6. Lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join tokens back into a single string
    return " ".join(cleaned_tokens)

# Apply the preprocessing function
# This line can take a few minutes to run
df['cleaned_review'] = df['review'].apply(preprocess_text)
print("Preprocessing complete!")

# Display the result to verify
print("\n--- Preprocessing Example ---")
print("Original Review:")
print(df['review'][0])
print("\nCleaned Review:")
print(df['cleaned_review'][0])

## ----------------------------------------------------------------
## Phase 4: Feature Extraction (Vectorization)
## ----------------------------------------------------------------
print("\nPhase 4: Vectorizing Text...")

# Encode the 'sentiment' column (positive=1, negative=0)
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])

# Initialize the TF-IDF Vectorizer
# max_features=5000 means it will only use the 5000 most frequent words
tfidf = TfidfVectorizer(max_features=5000)

# Create the feature matrix (X) and target vector (y)
X = tfidf.fit_transform(df['cleaned_review']).toarray()
y = df['sentiment_encoded']

print("Vectorization complete!")
print("Shape of feature matrix (X):", X.shape)
print("Shape of target vector (y):", y.shape)

## ----------------------------------------------------------------
## Phase 5: Model Building & Training
## ----------------------------------------------------------------
print("\nPhase 5: Splitting data and training model...")

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the Logistic Regression model
# max_iter is increased to ensure the model has enough iterations to converge
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model training complete!")

## ----------------------------------------------------------------
## Phase 6: Model Evaluation
## ----------------------------------------------------------------
