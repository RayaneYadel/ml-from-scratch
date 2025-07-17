# 🧠 Machine Learning from Scratch

This repository contains Python implementations of fundamental machine learning algorithms, coded entirely from scratch.  
No high-level libraries (like scikit-learn) were used for the core logic — only base Python and NumPy.

---

## 📁 File Overview

- `Classifiers.py` – Implementation of various classification algorithms
- `Clustering.py` – K-Means clustering algorithm
- `evaluation.py` – Accuracy, precision, recall, F1-score, and evaluation tools
- `utils.py` – Utility functions for preprocessing and general-purpose tools
- `__init__.py` – Makes the folder a Python package

---

## 🤖 Implemented Classifiers

From the file `Classifiers.py`:

- `ClassifierPerceptron` – Basic Perceptron without bias  
- `ClassifierPerceptronBiais` – Perceptron with bias  
- `ClassifierKNN` – K-Nearest Neighbors (binary classification)  
- `ClassifierKNN_MC` – KNN for multi-class classification  
- `ClassifierKNN_MC_bin` – Binary One-vs-One version of multi-class KNN  
- `ClassifierMultiOAA` – One-vs-All wrapper for multiclass classification  
- `ClassifierLineaireRandom` – Random linear classifier (baseline)  
- `ClassifierArbreDecision` – Decision Tree (categorical features)  
- `ClassifierArbreNumerique` – Decision Tree for numerical features  
- `ClassifierBaggingTree` – Bagging ensemble with decision trees

---

## 🔍 Features

- No external machine learning libraries used
- Modular and readable code structure
- Includes both **binary** and **multi-class** classifiers
- Educational and extensible design

---

## 🚀 How to Use

You can import and test the classifiers in your Python scripts or notebooks:

```python
from Classifiers import ClassifierKNN, ClassifierPerceptron
from evaluation import accuracy_score

