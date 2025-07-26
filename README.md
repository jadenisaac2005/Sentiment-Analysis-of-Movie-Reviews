# 🎬 Movie Review Sentiment Analysis Web App

A simple yet powerful web application that analyzes the sentiment of a movie review (Positive or Negative) using a machine learning model trained on the IMDb dataset. This project serves as a hands-on introduction to the end-to-end NLP project lifecycle, from data preprocessing to model deployment with a user-friendly interface.

![App Screenshot](https://i.postimg.cc/cCbZDwLZ/Screenshot-2025-07-27-at-1-30-38-AM.png)

---

## ✨ Features

- **Live Sentiment Prediction**: Get real-time sentiment analysis for any movie review text.
- **Confidence Score**: See the model's confidence in its prediction.
- **Clean UI**: A simple and intuitive user interface built with Streamlit.
- **End-to-End NLP Pipeline**: Demonstrates text cleaning, TF-IDF vectorization, and model training.
- **Baseline Model**: Uses a Logistic Regression model, achieving ~89% accuracy on the test set.

---

## 🛠️ Tech Stack

- **Python**: The core programming language.
- **Pandas**: For data manipulation and loading.
- **NLTK (Natural Language Toolkit)**: For text preprocessing tasks like tokenization, stopword removal, and lemmatization.
- **Scikit-learn**: For building and training the machine learning model (TF-IDF, Logistic Regression).
- **Streamlit**: To create and serve the interactive web application.
- **Joblib**: For saving and loading the trained model.

---

## 📂 Project Structure

The project is organized to separate the model training logic from the web application, which is a best practice for machine learning projects.

```
Sentiment-Analysis-of-Movie-Reviews/
│
├── data/
│   └── IMDB Dataset.csv      # The raw dataset
│
├── models/
│   ├── sentiment_model.joblib  # The saved trained model
│   └── tfidf_vectorizer.joblib # The saved TF-IDF vectorizer
│
├── app.py                      # The Streamlit web application script
├── train_model.py              # Script to train and save the model
└── README.md                   # You are here!
```

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Make sure you have Python 3.8+ installed on your system.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/jadenisaac2005/Sentiment-Analysis-of-Movie-Reviews.git](https://github.com/YOUR_USERNAME/Sentiment-Analysis-of-Movie-Reviews.git)
    cd Sentiment-Analysis-of-Movie-Reviews
    ```

2.  **Install the required Python libraries:**
    ```bash
    pip install pandas scikit-learn nltk streamlit joblib beautifulsoup4
    ```

3.  **Download NLTK data packages:**
    Run the following commands in a Python interpreter to download the necessary data for text processing.
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    ```

---

## 🖥️ Usage

The project has two main parts: training the model and running the application.

### 1. Train the Model

First, you need to run the training script. This will process the dataset and save the trained model and vectorizer into the `models/` directory.

```bash
python3 train_model.py
```
*This process may take a few minutes as it preprocesses 50,000 reviews.*

### 2. Run the Web App

Once the model is trained and saved, you can launch the Streamlit web application.

```bash
streamlit run app.py
```

Your web browser should automatically open to a local URL where the app is running. Now you can enter any movie review to see its predicted sentiment!

---

## 💡 Future Improvements

This project provides a solid foundation. Here are some ways it could be enhanced:

- **Try Different Models**: Experiment with `LinearSVC` or `MultinomialNB` to see if accuracy can be improved.
- **Build a Neural Network**: Implement a simple neural network using TensorFlow/Keras for potentially better performance.
- **Use Word Embeddings**: Replace TF-IDF with pre-trained word embeddings like Word2Vec or GloVe.
- **Deploy to the Cloud**: Deploy the Streamlit app to a service like Streamlit Community Cloud or Heroku to make it publicly accessible.

---

## 📬 Contact

Jaden Isaac – A B.Tech AI & ML student passionate about building useful projects and exploring the world of technology.

Feel free to reach out with any questions or feedback!

- **GitHub**: [github.com/jadenisaac2005](https://github.com/jadenisaac2005)
- **LinkedIn**: [linkedin.com/in/jaden-isaac](https://linkedin.com/in/jaden-isaac)
