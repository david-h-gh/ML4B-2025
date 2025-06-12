import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Modell und LabelEncoder laden
model = load_model("hundemodell.h5")
with open("labelencoder.pkl", "rb") as f:
    le = pickle.load(f)

# Bild vorbereiten: Bild laden, auf RGB setzen, Größe anpassen, Normalisierung
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")  # In RGB konvertieren
    img = img.resize((224, 224))                    # Größe an Model-Anforderung anpassen
    img_array = np.array(img) / 255.0               # Pixelwerte auf 0-1 skalieren
    return img_array.reshape(1, 224, 224, 3)        # Batch-Dimension hinzufügen

# Streamlit UI-Titel
st.title("🐶 Hunderassen-Erkenner")

# Datei-Upload Widget: Nur Bilder akzeptieren
uploaded_file = st.file_uploader("📤 Lade ein Bild hoch", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild anzeigen
    st.image(uploaded_file, caption="📷 Hochgeladenes Bild", use_column_width=True)

    # Bild vorbereiten für das Modell
    img = preprocess_image(uploaded_file)
    
    # Modell-Vorhersage (Wahrscheinlichkeiten für jede Klasse)
    prediction = model.predict(img)[0]  # [0], weil wir nur ein Bild haben, nicht Batch
    
    # Wahrscheinlichkeiten mit den Klassennamen aus LabelEncoder verknüpfen
    proba_df = pd.DataFrame({
        "Rasse": le.classes_,                    # Klassen-Namen (Hunderassen)
        "Wahrscheinlichkeit (%)": prediction * 100  # Wahrscheinlichkeiten in Prozent
    }).sort_values("Wahrscheinlichkeit (%)", ascending=False)  # Absteigend sortieren
    
    # Beste Vorhersage (erste Zeile, da sortiert)
    top_rasse = proba_df.iloc[0]

    # Schwellenwert prüfen: mindestens 30% Wahrscheinlichkeit für "Hund erkannt"
    if top_rasse["Wahrscheinlichkeit (%)"] < 30:
        st.subheader("⚠️ Kein Hund erkannt (Wahrscheinlichkeit < 30%)")
    else:
        st.subheader(f"✅ Vorhergesagte Rasse: **{top_rasse['Rasse']}**")
        st.write(f"Mit einer Wahrscheinlichkeit von **{top_rasse['Wahrscheinlichkeit (%)']:.2f}%**")

        # Kreisdiagramm der Top 5 Wahrscheinlichkeiten als Visualisierung
        st.subheader("📊 Wahrscheinlichkeitsverteilung (Top 5)")
        top5 = proba_df.head(5)
        fig, ax = plt.subplots()
        ax.pie(top5["Wahrscheinlichkeit (%)"], labels=top5["Rasse"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")  # Kreisförmig machen
        st.pyplot(fig)

        # Optional: gesamte Wahrscheinlichkeitsverteilung als Tabelle anzeigen (ausklappbar)
        with st.expander("🔍 Alle Wahrscheinlichkeiten anzeigen"):
            st.dataframe(proba_df.reset_index(drop=True).round(2))
