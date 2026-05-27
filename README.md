# 🧠 Heart Disease Classification From Scratch

A machine learning project implementing classical ML algorithms from scratch without using scikit-learn.

---

# 📌 Project Overview

This project aims to classify heart disease cases using handcrafted machine learning algorithms and a fully modular ML pipeline.

Implemented algorithms:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree

The project includes:
- data preprocessing
- evaluation metrics
- experiment tracking
- visualization
- comparative analysis

---

# 📂 Project Structure

```bash
project/
│
├── data/
│
├── figures/
│   ├── data_distribution.png
│   ├── feature_analysis.png
│   ├── loss_curves.png
│   └── model_comparison.png
│
├── notebooks/
│   ├── experiment.ipynb
│   └── exploration.ipynb
│
├── src/
│   ├── classical_models.py
│   ├── preprocessing.py
│   ├── metrics.py
│   └── utils.py
│
├── experiment_log.txt
├── metrics_table.csv
├── requirements.txt
└── README.md
```

---

# ⚙️ Implemented Models

## Logistic Regression
Implemented using:
- gradient descent
- sigmoid activation
- binary cross-entropy loss

---

## K-Nearest Neighbors (KNN)
Implemented using:
- Euclidean distance
- majority voting

---

## Decision Tree
Implemented using:
- entropy
- information gain
- recursive tree building

---

# 📊 Evaluation Metrics

Implemented from scratch:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# 📈 Visualizations

Generated figures:
- Data distribution
- Feature correlation analysis
- Logistic regression loss curves
- Model comparison charts

---

# 🚀 Results

Example results obtained:

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 0.97 | 0.96 |
| KNN | 0.95 | 0.94 |
| Decision Tree | 0.93 | 0.91 |

Logistic Regression achieved the best overall performance on the dataset.

---

# ▶️ How to Run

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run experiments

Open:

```bash
notebooks/experiment.ipynb
```

or execute experiment scripts directly.

---

# 🧪 Future Improvements

- Cross-validation
- Feature scaling
- Hyperparameter tuning
- Decision tree pruning
- Additional ML models

---

# 📚 Educational Objective

This project was developed for educational purposes to better understand the internal mechanics of machine learning algorithms without relying on scikit-learn implementations.