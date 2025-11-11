import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    cleaned_words = []
    for word in text:
        if word.isalnum():
            cleaned_words.append(word)
    final_words = []
    for word in cleaned_words:
        if word not in string.punctuation and word not in stopwords.words('english'):
            final_words.append(word)
    final_words = [ps.stem(word) for word in final_words]
    return " ".join(final_words)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")