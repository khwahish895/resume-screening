import streamlit as st
import pickle
import re
import PyPDF2

# Load model and vectorizer
model = pickle.load(open("clf.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.set_page_config(page_title="Resume Screening App", layout="centered")

st.title("Resume Screening App")
st.write("Upload a resume PDF to predict the job category")

# ---------- Text cleaning ----------
def clean_text(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^A-Za-z ]+', ' ', text)
    return text.lower()

# ---------- PDF text extraction ----------
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ---------- File upload ----------
uploaded_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])

if uploaded_file is not None:

    # File size check (bytes)
    if uploaded_file.size < 2000:
        st.error("Uploaded file is too small")
    else:
        try:
            resume_text = extract_text_from_pdf(uploaded_file)

            # ✅ Show extracted text (optional)
            if st.checkbox("Show extracted text"):
                st.text_area(
                    "Extracted Resume Text",
                    resume_text,
                    height=300
                )

            # Text length check
            if len(resume_text.strip()) < 100:
                st.error("Resume text is too short or unreadable")
            else:
                cleaned_text = clean_text(resume_text)

                # Vectorize
                vector = tfidf.transform([cleaned_text])

                # Predict
                prediction = model.predict(vector)[0]

                st.success("Resume processed successfully")
                st.subheader("Predicted Category")
                st.write(prediction)

        except Exception as e:
            st.error(f"Error processing the file: {e}")
