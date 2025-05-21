import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Modell laden
#model = tf.keras.models.load_model("hundemodell.h5")

# Label-Index-Mapping
#label_map = {0: 'Golden Retriever', 1: 'Pudel', 2: 'Schäferhund'}  # Beispiel

# Streamlit UI
st.title("Hunderasse erkennen")
uploaded_file = st.file_uploader("Lade ein Hundebild hoch", type=["jpg", "png", "jpeg"])

#if uploaded_file is not None:
#    image = Image.open(uploaded_file)
#    st.image(image, caption='Hochgeladenes Bild', use_column_width=True)

    # Bild vorbereiten
#    img = image.resize((224, 224))  # gleiche Größe wie beim Training
#    img_array = np.array(img) / 255.0
#    img_array = np.expand_dims(img_array, axis=0)

    # Vorhersage
#    prediction = model.predict(img_array)
#    predicted_class = label_map[np.argmax(prediction)]

#   st.subheader(f"Vorhersage: {predicted_class}")