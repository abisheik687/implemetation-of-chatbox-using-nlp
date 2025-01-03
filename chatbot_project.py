
import os
import json
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Download necessary NLTK data
nltk.download('punkt')

# Load intents data
file_path = 'intents.json'  # Ensure this file is in the same directory as the script
with open(file_path, 'r') as file:
    intents = json.load(file)

# Prepare data for training
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Train the model
X = vectorizer.fit_transform(patterns)
y = tags
clf.fit(X, y)

# Define chatbot function
def chatbot(input_text):
    input_text_vectorized = vectorizer.transform([input_text])
    predicted_tag = clf.predict(input_text_vectorized)[0]
    for intent in intents:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])

# Streamlit Deployment
def main():
    st.title('Intent-Based Chatbot')
    st.write('Welcome! Type your message below to chat.')

    user_input = st.text_input('You:')
    if user_input:
        response = chatbot(user_input)
        st.text_area('Chatbot:', value=response, height=100, max_chars=None)

if __name__ == '__main__':
    main()
