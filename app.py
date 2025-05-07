import streamlit as st
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

st.title("Fake News Detector 🕵️‍♂️")
text_input = st.text_area("Masukkan berita atau pernyataan in English:")

if st.button("Deteksi"):
    if text_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        label, confidence = predict(text_input)
        st.success(f"Prediksi: **{label}** ({confidence}% yakin)")
