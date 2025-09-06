# physio-pain-predictor
An AI-powered system to predict pain levels using physiological sensor data and machine learning techniques
Predicred pain levels (Pain / No Pain) in subjects based on their physiological signals. Using machine learning techniques, specifically Support Vector Machine (SVM), the system can classify whether a subject is experiencing pain using measurements like blood pressure, respiration rate, and electrodermal activity.

The goal is to provide automated pain assessment that can be useful in healthcare monitoring, wearable devices, or patient care systems where verbal feedback may not be possible.


**Dataset**

The dataset contains physiological measurements collected from multiple subjects. Key features include:

BP Dia_mmHg – Diastolic Blood Pressure

LA Systolic BP_mmHg – Systolic Blood Pressure

EDA_microsiemens – Electrodermal Activity

Respiration Rate_BPM – Breathing Rate

Class – Pain or No Pain

Each record include computed statistics like mean, variance, min, max for the physiological signals.

**Features**

Preprocess physiological data and compute summary statistics.

Convert class labels into numeric form: No Pain = 0, Pain = 1.

Option to filter by physiological signal type or use all available features.

Train a Support Vector Machine (SVM) classifier.

Perform 10-fold cross-validation for robust performance evaluation.

Generate metrics including accuracy, precision, recall, and confusion matrix.
