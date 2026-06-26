# 🚢 Titanic: ML from Disaster (Kaggle)

An object-oriented Machine Learning model built to predict passenger survival rates for the Kaggle Titanic dataset, built as my first end-to-end data science project for the Kaggle Titanic Challenge.


## ✨ Features
* **Stratified Validation:** Uses StratifiedShuffleSplit on passenger class (Pclass) to ensure training and testing splits perfectly mirror real-world survival distributions.

* **Custom End-to-End Pipeline:**  Implements a strict **Scikit-Learn Pipeline** using object-oriented custom transformers to prevent data leakage.

* **Smart Imputation:** Features an automated AgeImputer that dynamically handles missing values using a mean imputation strategy metric.

* **One-Hot Encoding Matrix:** Hand-builds encoded matrices for categorical elements (Sex, Embarked) with explicit safety drop parameters.

* **Optimized Aesthetics:**  Global configuration settings configured via sns.set_theme and rcParams for easy-to-read, high-definition visualization analysis.

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Data Analytics:** Pandas, Numpy
* **Data Visualization:** Seaborn, Matplotlib
* **Machine Learning Engine:** Scikit-Learn (Pipeline, RandomForestClassifier, GridSearchCV)
* **Core Estimator:** Ensemble Random Forest Classifier

## 🏗️ Data Pipeline Architecture

```
[Raw DataFrame] 
       │
       ▼
 ┌───────────────┐
 │  AgeImputer   │ ──► Handles missing values using "mean" strategy
 └───────────────┘
       │
       ▼
 ┌───────────────┐
 │FeatureEncoder │ ──► Converts "Sex" and "Embarked" into structural binary matrices
 └───────────────┘
       │
       ▼
 ┌───────────────┐
 │FeatureDropper │ ──► Drops unneeded identifiers safely ("Cabin", "Ticket", "Name", etc.)
 └───────────────┘
       │
       ▼
 [Processed Array] ──► Fed directly into StandardScaler & RandomForestClassifier
```

## 🚀 Setup

1. Clone the repo.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn` 
3. Download titanic.csv and test.csv from Kaggle and place them in the project folder.
4. Run Titanic.ipynb cell by cell.
5. Find your predictions in Predictions.csv!
