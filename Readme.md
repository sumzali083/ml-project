
# Stock Price Movement Prediction (BlackRock)

## Overview

This project explores whether machine learning can predict **next-day stock price direction** using historical market data.

Using historical stock data for **BlackRock (BLK)**, we build a machine learning pipeline that:

* Collects and processes stock market data
* Engineers financial features
* Trains classification models
* Evaluates predictive performance

The goal is to predict whether the **closing price of the next trading day will be higher or lower than today**.

---

# Dataset

The dataset contains historical stock data for **BlackRock (BLK)** including:

* Open price
* High price
* Low price
* Close price
* Trading volume

The dataset was downloaded using the **Yahoo Finance API**.

Each row represents one trading day.

---

# Feature Engineering

Several financial features were created from the raw data to capture market behaviour.

### Volatility

Daily price range:

```
Volatility = High − Low
```

Measures how much the price fluctuates during a trading day.

### Daily Return

Percentage change in closing price:

```
Daily Return = Close.pct_change()
```

Captures short-term momentum.

### Volume Change

Percentage change in trading volume:

```
Volume Change = Volume.pct_change()
```

Measures changes in market activity.

### Moving Average (MA5)

Five-day moving average of the closing price:

```
MA5 = rolling mean of Close over 5 days
```

Helps capture short-term trends.

---

# Prediction Target

The model predicts **whether the stock price will increase the next day**.

```
Target = 1 if tomorrow_close > today_close
Target = 0 otherwise
```

This transforms the problem into a **binary classification task**.

---

# Machine Learning Pipeline

The following steps were used in the project:

1. Data collection
2. Data inspection and cleaning
3. Feature engineering
4. Target variable creation
5. Removal of missing values
6. Feature/target separation
7. Train-test split
8. Feature scaling
9. Model training
10. Model evaluation

---

# Models Used

Three machine learning models were tested:

### Logistic Regression

A simple baseline classification model commonly used in financial modelling.

### K-Nearest Neighbors (KNN)

A distance-based model that predicts outcomes based on similar historical market conditions.

### Random Forest

---

# Evaluation Metrics

Model performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC Curve
* ROC AUC Score

---

# Results

Both models produced performance close to **random guessing** (ROC AUC ≈ 0.50).

This result highlights an important reality:

> Short-term stock price movements are extremely difficult to predict using only historical price and volume data.

Financial markets contain a large amount of noise and are influenced by external factors such as news, macroeconomic events, and institutional trading.

---

# Key Takeaways

* Building the **machine learning pipeline** is the main objective of the project.
* Feature engineering plays a critical role in financial modelling.
* Simple price-based features alone provide limited predictive power.
* Even small predictive signals can be valuable in quantitative finance.

---

# Project Structure

```
ML-PROJECT/
│
├── data/
│   └── blackrock_stock.csv
│
├── notebooks/
│   └── exploration.ipynb
│
├── models/
│
├── src/
│
└── README.md
```

---

# Future Improvements

Potential improvements to the project include:

* Adding more financial indicators such as:

  * RSI (Relative Strength Index)
  * MACD
  * Bollinger Bands
* Testing additional models such as:

  * Gradient Boosting
  * XGBoost
* Performing hyperparameter tuning
* Expanding the dataset to include multiple stocks

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---
