import gradio as gr
import numpy as np
import joblib
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Load the pre-trained Word2Vec model
w2v_model = Word2Vec.load("word2vec.model")

# Load the pre-trained SVM model
svm_classifier = joblib.load("best_model.joblib")

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def predict_sentiment(text):
    # Preprocess and create Word2Vec vectors from the input text
    tokens = preprocess_text(text)
    text_vector = np.mean([w2v_model.wv[word] for word in tokens.split() if word in w2v_model.wv], axis=0)
    if not np.any(text_vector):
        return "Cannot determine sentiment due to missing words in vocabulary."
    # Make prediction using the pre-trained SVM classifier
    prediction = svm_classifier.predict([text_vector])[0]
    return "Positive" if prediction == "positive" else "Negative"

# Create the Gradio interface
gr.Interface(
    fn=predict_sentiment,
    inputs="text",
    outputs="text",
    title="Movie Sentiment Analysis",
    description="Predicts the sentiment of a movie review as positive or negative."
).launch(share=True)
