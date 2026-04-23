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
├── stress\\\_predictor.ipynb   # Main Jupyter Notebook (EDA + training + evaluation)
├── stress\\\_predictor.py      # Python script version
├── mental\\\_health.csv        # Training dataset
├── unseen\\\_data.csv          # New data for prediction
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

\---

## 🔧 Features Used

After data cleaning, the following features were selected:

* `Age`
* `Sleep\\\_Hours`
* `Work\\\_Study\\\_Hours`
* `Physical\\\_Activity` *(encoded: Low / Medium / High)*
* `Social\\\_Interaction\\\_Score`
* `Burnout`
* `Depression`
* `Night\\\_Usage`
* `Occupation` *(one-hot encoded)*

**Target variable:** `Stress\\\_Level`

**Dropped columns:** `Person\\\_ID`, `Gender`, `Anxiety`, `Alcohol`, `Smoking`, `Caffeine\\\_Intake`

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
jupyter notebook stress\\\_predictor.ipynb
```

Or run the Python script directly:

```bash
python stress\\\_predictor.py
```

> ⚠️ Make sure `mental\\\_health.csv` and `unseen\\\_data.csv` are in the same folder before running.

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

The trained Random Forest model is applied to `unseen\\\_data.csv`. A new column `Predicted stress level` is added to the output showing the predicted stress scores.

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

