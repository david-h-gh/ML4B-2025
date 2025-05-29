import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pickle

# Modell und LabelEncoder laden
model = load_model("hundemodell.h5")
with open("labelencoder.pkl", "rb") as f:
    le = pickle.load(f)

# Bild vorbereiten
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 224, 224, 3)

# Streamlit UI
st.title("Hunderassen-Erkenner")

uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Hochgeladenes Bild", use_column_width=True)

    img = preprocess_image(uploaded_file)
    prediction = model.predict(img)
    predicted_class = le.inverse_transform([np.argmax(prediction)])[0]

    st.subheader("Vorhergesagte Rasse:")
    st.write(predicted_class)