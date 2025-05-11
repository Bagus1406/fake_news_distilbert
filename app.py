import streamlit as st
import streamlit.components.v1 as stc
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load model dari Hugging Face Hub
model_path = "kupr0y/fake-news-model" 

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# HTML Header
html_temp = """<div style="background-color:#2c3e50;padding:10px;border-radius:10px">
                <h1 style="color:white;text-align:center">Fake News Detector</h1> 
                <h4 style="color:white;text-align:center">Built with Hugging Face & Streamlit</h4> 
              </div>"""

desc_temp = """ ### Welcome!
This app uses a fine-tuned DistilBERT model to classify news as **FAKE** or **REAL**.
#### Model Source
[Hugging Face: kupr0y/fake-news-model](https://huggingface.co/kupr0y/fake-news-model)
"""

# Prediksi fungsi
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1).item()
    
    label = "FAKE" if pred == 0 else "REAL"
    confidence = round(prob[0][pred].item() * 100, 2)
    return label, confidence

# Main App
def main():
    stc.html(html_temp)

    menu = ["Home", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)

    elif choice == "Prediction":
        st.subheader("Fake News Prediction")

        text_input = st.text_area("Masukkan berita atau pernyataan (dalam Bahasa Inggris):")
        if st.button("Deteksi"):
            if text_input.strip() == "":
                st.warning("Teks tidak boleh kosong.")
            else:
                label, confidence = predict(text_input)
                if label == "REAL":
                    st.success(f"Prediksi: **{label}** ({confidence}% yakin)")
                else:
                    st.error(f"Prediksi: **{label}** ({confidence}% yakin)")

if __name__ == "__main__":
    main()
