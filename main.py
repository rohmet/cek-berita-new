import streamlit as st
import joblib
import re

# ==============================
#  Preprocessing Function
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================
#  Load Model & Vectorizer
# ==============================
@st.cache_resource
def load_all():
    vectorizer = joblib.load("vectorizer.pkl")
    model_nb = joblib.load("model_nb.pkl")
    model_svm = joblib.load("model_svm.pkl")
    model_voting = joblib.load("model_voting.pkl")
    return vectorizer, model_nb, model_svm, model_voting

vectorizer, model_nb, model_svm, model_voting = load_all()


# ==============================
#  Streamlit UI
# ==============================
st.set_page_config(page_title="News Classification Indonesia", layout="centered")

st.title("📰 Klasifikasi Berita Indonesia")
st.write("""
Aplikasi ini memprediksi kategori berita berdasarkan **judul + isi berita** menggunakan 
tiga model Machine Learning (Naive Bayes, SVM, Voting Classifier).  
Dataset: *News Indonesia Kaggle*.  
""")


# ==============================
#  Sidebar Info
# ==============================
st.sidebar.title("📊 Model Performance")

# Masukkan akurasi dari hasil training kamu (ganti nilai di bawah ini)
acc_nb = 0.0   # ganti sesuai hasil kamu
acc_svm = 0.0
acc_vote = 0.0

st.sidebar.write(f"**Naive Bayes Accuracy:** {acc_nb*100:.2f}%")
st.sidebar.write(f"**SVM Accuracy:** {acc_svm*100:.2f}%")
st.sidebar.write(f"**Voting Accuracy:** {acc_vote*100:.2f}%")


st.sidebar.write("---")
st.sidebar.write("Made by Rohmet")


# ==============================
#  Input Form
# ==============================
st.header("Masukkan Berita Baru")

title = st.text_input("Judul Berita")
content = st.text_area("Isi Berita")

model_choice = st.radio(
    "Pilih Model Prediksi",
    ("Naive Bayes", "SVM", "Voting Classifier")
)


# ==============================
#  Prediction Button
# ==============================
if st.button("🔍 Prediksi Kategori Berita"):
    if title == "" and content == "":
        st.warning("Masukkan minimal judul atau isi berita!")
    else:
        # preprocessing
        combined = title + " " + content
        cleaned = clean_text(combined)
        X_input = vectorizer.transform([cleaned])

        # pilih model
        if model_choice == "Naive Bayes":
            pred = model_nb.predict(X_input)[0]
        elif model_choice == "SVM":
            pred = model_svm.predict(X_input)[0]
        else:
            pred = model_voting.predict(X_input)[0]

        st.success(f"**Kategori Berita: {pred}**")
