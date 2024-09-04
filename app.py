
import pickle
import string
import nltk
import spacy
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize the PorterStemmer
ps = PorterStemmer()

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Define the set of stopwords
stop_words = set(stopwords.words('english'))

def transform_text_spacy(text):
    # Process the text with spaCy
    doc = nlp(text.lower())

    # Initialize an empty list to store tokens
    y = []

    # Iterate through the tokens in the doc
    for token in doc:
        # Check if the token is alphanumeric (removes special characters)
        if token.text.isalnum():
            y.append(token.text)

    text = y[:]
    y.clear()

    # Remove stop words and punctuation
    for token in text:
        if token not in stop_words and token not in string.punctuation:
            y.append(token)

    text = y[:]
    y.clear()

    # Apply stemming
    for token in text:
        y.append(ps.stem(token))

    return " ".join(y)

# Load the trained TF-IDF vectorizer and model
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Set the title of the Streamlit app
st.title("EMAIL SPAM CLASSIFIER")

# Get input from the user
input_Email = st.text_input("Enter the Email")

# Check if the Predict button is clicked
if st.button("Predict"):
    # 1. Preprocess the input email
    transform_Email = transform_text_spacy(input_Email)

    # 2. Vectorize the preprocessed email
    vector_input = tfidf.transform([transform_Email])

    # 3. Predict if the email is spam or not
    result = model.predict(vector_input)[0]

    # 4. Display the result
    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")
