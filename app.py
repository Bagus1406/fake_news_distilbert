import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Inisialisasi model dan tokenizer dari model
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("kupr0y/fake-news-model")
    model = DistilBertForSequenceClassification.from_pretrained("kupr0y/fake-news-model")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Fungsi prediksi
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prob = torch.softmax(logits, dim=1)
        label_id = torch.argmax(prob, dim=1).item()
    labels_map = {0: "FAKE", 1: "REAL"}
    label = labels_map[label_id]
    confidence = round(prob[0][label_id].item() * 100, 2)
    return label, confidence

# Tampilan UI Streamlit
def main():
    st.markdown(
        """<div style="background-color:#2c3e50;padding:10px;border-radius:10px">
            <h1 style="color:white;text-align:center">Fake News Detector</h1> 
            <h4 style="color:white;text-align:center">Built with Hugging Face & Streamlit</h4> 
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """### Welcome!
This app uses a fine-tuned DistilBERT model to classify news as **FAKE** or **REAL**.  
#### Model Source  
[Hugging Face: kupr0y/fake-news-model](https://huggingface.co/kupr0y/fake-news-model)
""",
        unsafe_allow_html=True,
    )

    menu = ["Home", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Prediction":
        st.subheader("Fake News Prediction")
        user_input = st.text_area("Masukkan berita atau pernyataan (dalam Bahasa Inggris):")
        if st.button("Deteksi"):
            if user_input.strip() == "":
                st.warning("Teks tidak boleh kosong.")
            else:
                label, confidence = predict(user_input)
                if label == "REAL":
                    st.success(f"Prediksi: **{label}** ({confidence}% yakin)")
                else:
                    st.error(f"Prediksi: **{label}** ({confidence}% yakin)")
    else:
        st.subheader("Home")

if __name__ == "__main__":
    main()
