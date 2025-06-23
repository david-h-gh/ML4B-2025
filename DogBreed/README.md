1 Introduction
ERkennung von Hunderassenerkennung hat praxisorientierte Anwendungsfälle. Ein Anwendungsfall ist z.B. in Tierheimen oder als App für Hundebesitzer. Außerdem dient es als erster Schritt, sich mit dem Erstellen von ML-Modellen zu beschäftigen.
Research Question: Wie kann ein Deep-Learning-Modell trainiert werden, um auf Grundlage von Hundebildern zuverlässig die jeweilige Hunderasse zu identifizieren?

2 Related Work 
Wir haben uns an dem Datensatz von Kaggle orientiert. Hier wurde bereits eine Competition veranstaltet, in der es darum ging, das best mögliche Modell für die Hunderasse Erkennung zu erstellen.
Außerdem gibt es viele andere Beispiele von anderen Bilderkennungsmodellen, welche wir auch auf unseren Fall anpassen können.

3 Methodology https://www.scrapy.org/

3.1 General Methodology 
1. Auswahl und Download eines Datensatzes. Wir haben uns für den Datensatz von Kaggle entschieden
2. Datenbereinigung und -aufbereitung. 
3. Training des Modells mit sklearn und tenserflow.
4. Evaluation unseres Modells. Erstellung eines Confusion Matrix um zu erkennen, wie gut die Rassen erkannt werden.(Hier haben wir bis jetzt noch nicht so viel gemacht)
5. Entwicklung einer Anwendung zur Klassifikation in Streamlit

3.2 Data Understanding and Preparation
Generell ist der Datenschatz schon gut gepflegt.
Es gibt 2 Datensätze mit jeweils ca 10.000 Bildern. Einer ist für das Training und einer für den Test des Modells. Der Tainingsdatensatz hat auch noch eine csv-Datei, welche zu jedem Bild die Rasse angibt.
Wir haben hier vor allem die Trainingsbilder genormt und dabei alle Bilder auf 224x224 Pixel formatiert und die verarbeiteten Bidler neu abgespeichert.

3.3 Modeling and Evaluation 
Für das Training wurde sklearn und tesnerflow verwendet. 
Parameter:
- BATCH_SIZE = 32
- LEARNING_RATE = 0.001
- DROPOUT_RATE = 0.7
Evaluation:
- Metriken: Confusion Matrix

4 Results Describe
Artefakte:
- Ein trainiertes Modell zur Hunderassenaerkennung
- Eine Anwendung, die neue Bilder klassifiziert
Bibiliothek:
- Python
- TensorFlow
- numpy
- matplotlib
- seaborn
- sklearn
Konzept der App:
- User lädet Bild hoch und kriegt eine Einschätzung welche Hunderasse das Bild zeigt
- Außerdem Aufteilung der 5 wahrscheinlichsten Hunderassen, die das Bild zeigen könnte
Ergebnisse der Daten:
- accuracy: 0.8575
- loss: 0.5690

5 Discussion Now its time to discuss your results/ artifacts/ app Show the limitations : e.g. missing data, limited training ressources/ GPU availability in Colab, limitaitons of the app Discuss your work from an ethics perspective: Dangers of the application of your work (for example discrimination through ML models) Transparency Effects on society and environment Possible sources https://algorithmwatch.org/en/ Have a look at the "Automating Society Report"; https://ainowinstitute.org/ Have a look at this website and their publications Further Research: What could be next steps for other researchers (specific research questions)
