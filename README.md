# 🧠 Stress Level Predictor

A machine learning project that predicts an individual's **stress level** based on lifestyle and behavioral factors such as sleep hours, work/study hours, physical activity, and social interaction.

\---

## 📌 Project Overview

This project compares three regression models to find the best predictor of stress level:

|Model|Description|
|-|-|
|Random Forest Regressor|Ensemble of decision trees|
|Linear Regression|Baseline linear model|
|Decision Tree Regressor|Single tree with max depth = 2|

The best-performing model (based on **Mean Absolute Error**) is used to make predictions on unseen data.

\---

## 📂 Project Structure

```
stress-predictor/
│
├── stress_predictor.ipynb   # Main Jupyter Notebook (EDA + training + evaluation)
├── stress_predictor.py      # Python script version
├── mental_health.csv        # Training dataset
├── unseen_data.csv          # New data for prediction
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

\---

## 🔧 Features Used

After data cleaning, the following features were selected:

* `Age`
* `Sleep_Hours`
* `Work_Study_Hours`
* `Physical_Activity` *(encoded: Low / Medium / High)*
* `Social_Interaction_Score`
* `Burnout`
* `Depression`
* `Night_Usage`
* `Occupation` *(one-hot encoded)*

**Target variable:** `Stress_Level`

**Dropped columns:** `Person_ID`, `Gender`, `Anxiety`, `Alcohol`, `Smoking`, `Caffeine_Intake`

\---

## ⚙️ How to Run

### 1\. Clone the repository

```bash
git clone https://github.com/GenX-Labs/stress-predictor.git
cd stress-predictor
```

### 2\. Install dependencies

```bash
pip install -r requirements.txt
```

### 3\. Run the notebook

```bash
jupyter notebook stress_predictor.ipynb
```

Or run the Python script directly:

```bash
python stress_predictor.py
```

> ⚠️ Make sure `mental_health.csv` and `unseen_data.csv` are in the same folder before running.

\---

## 📊 Model Evaluation

Models are evaluated using **Mean Absolute Error (MAE)** — lower is better.

```
RandomForestRegressor  → moderate MAE
LinearRegression       → moderate MAE
DecisionTreeRegressor  → lowest MAE (chosen for deployment)
```

\---

## 🚀 Prediction on New Data

The trained Random Forest model is applied to `unseen_data.csv`. A new column `Predicted stress level` is added to the output showing the predicted stress scores.

\---

## 🛠️ Tech Stack

* Python 3.x
* pandas
* scikit-learn
* matplotlib
* Jupyter Notebook

\---

## 👤 Author

**Rugendran**

* GitHub: https://github.com/GenX-Labs/

\---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

