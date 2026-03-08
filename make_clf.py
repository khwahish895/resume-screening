import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load dataset
df = pd.read_csv("UpdatedResumeDataSet.csv")

def clean_text(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^A-Za-z ]+', ' ', text)
    return text.lower()

df["clean_resume"] = df["Resume"].apply(clean_text)

X = df["clean_resume"]
y = df["Category"]

# Vectorizer
tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
X_tfidf = tfidf.fit_transform(X)

# Model
model = LinearSVC()
model.fit(X_tfidf, y)

# Save files
pickle.dump(model, open("clf.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

print("clf.pkl created successfully")
