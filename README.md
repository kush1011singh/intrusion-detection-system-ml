# Intrusion Detection System (IDS) using Machine Learning


## 📌 Overview
This project implements a **Machine Learning-based Intrusion Detection System (IDS)** trained on the **NSL-KDD dataset** to detect and classify different types of cyber attacks.  
It leverages **feature selection, data preprocessing, and multiple ML algorithms** to identify network intrusions with high accuracy.

The IDS is designed for **research, educational, and demonstration purposes** and showcases the application of **ML in cybersecurity**.

---

## 🎯 Objectives
- Build a robust IDS using ML techniques.
- Train on the **NSL-KDD** dataset for real-world attack detection.
- Compare the performance of multiple ML algorithms.
- Visualize attack patterns and feature correlations.
- Store trained models for easy deployment.

---

## 📂 Project Structure
IDS-ML/
│
├── models/ # Saved trained ML models (.joblib)
├── images/ # Visualizations and confusion matrices
├── ids_ml.py # Main project code
├── KDDTrain+.txt # Training dataset (NSL-KDD)
├── KDDTest+.txt # Testing dataset (NSL-KDD)
├── README.md # Project documentation
└── requirements.txt # Python dependencies

markdown
Copy
Edit

---

## 📊 Dataset
**Dataset:** NSL-KDD  
- Download: [https://www.unb.ca/cic/datasets/nsl.html](https://www.unb.ca/cic/datasets/nsl.html)  
- **Total Features:** 41 + label
- **Attack Categories:**
  - **DoS (Denial of Service)**
  - **Probe**
  - **R2L (Remote to Local)**
  - **U2R (User to Root)**
  - **Normal**

---

## ⚙️ Features
- **Data Preprocessing**
  - One-hot encoding for categorical features.
  - Feature scaling with StandardScaler.
  - Feature selection using SelectKBest.
- **ML Models Implemented**
  - Random Forest Classifier
  - Support Vector Machine (SVM)
- **Evaluation**
  - Accuracy, Precision, Recall, F1-Score.
  - Confusion Matrices.
- **Visualization**
  - Attack category distribution.
  - Protocol type distribution.
  - Feature correlation heatmap.

---

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/ids-ml.git
cd ids-ml
```
2️⃣ Install Dependencies
```bash
pip install -r requirements.txt

```
3️⃣ Download Dataset
Place KDDTrain+.txt and KDDTest+.txt in the project directory.

4️⃣ Run the IDS
```bash

python ids_ml.py
```
📈 Results
Model	Accuracy	Precision	Recall	F1-Score
Random Forest	99.1%	99%	99%	99%
SVM	97.8%	98%	98%	98%

Visualizations:

Attack category distribution (images/attack_distributions.png)

Protocol type distribution

Correlation matrix (images/feature_correlations.png)

Confusion matrices for each model


🛡 Disclaimer

This project is intended for educational and research purposes only.
It should not be used in production environments without thorough testing and compliance checks.
