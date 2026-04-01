# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
#  **full ML pipeline**.
# 
# 1. Collect the dataset (stock price data)
# 2. Load the dataset into a dataframe
# 3. Inspect the dataset (`head`, `info`, `describe`)
# 4. Understand the meaning of each column (Open, High, Low, Close, Volume)
# 5. Clean the data
# 6. Explore the data (EDA)
# 7. Engineer features
#    * volatility (High − Low)
#    * daily return
#    * volume change
#    * moving averages
# 8. Add the engineered features to the dataframe
# 9. Define the prediction target
#    * compare tomorrow's close to today's close
#    * convert to binary target (1 = up, 0 = down)
# 10. Handle NaN rows created by rolling windows and shifts
# 11. Remove rows that contain NaN values
# 12. Separate features and target
# * **X = feature columns**
# * **y = target column**
# 13. Split the dataset into training and testing sets
# 14. Choose a machine learning model
# 15. Train the model using the training data
# 16. Make predictions on the test data
# 17. Evaluate model performance (accuracy, precision, recall, etc.)
# 18. Improve the model
# * try new features
# * try different models
# * tune parameters
# 19. Save the trained model
# 20. Use the model to predict future stock movements
# 

# %%
import yfinance as yf

data = yf.download("BLK", start="2010-01-01")

# %%
data.to_csv("../data/blackrock_stock.csv")

# %%
print(data.head())
print(data.info())
print(data.describe())

# %% [markdown]
# cleaning the data 

# %%
df = data.dropna()
df.drop_duplicates(inplace=True)
valid_numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
df.dtypes

print(df.head())

# %% [markdown]
# create features 

# %%
volatility = df['High'] - df['Low']
print(volatility.head())
daily_return = df['Close'].pct_change()
print(daily_return.head())
volume_change = df['Volume'].pct_change()
print(volume_change.head())
five_day_average = df['Close'].rolling(window=5).mean()
print(five_day_average.head())


# %%
df["Volatility"] = volatility
df["Daily Return"] = daily_return
df["Volume Change"] = volume_change
df["five day average"] = five_day_average
df.head()


# %% [markdown]
# Prediction target 
# using binary classification for simplicity 

# %%
# correct the target column with a vectorised calculation
# 1 if next‑day close is higher than today, 0 otherwise
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# the last row gets NaN from the shift, make it 0 (or drop it later)
df['Target'].fillna(0, inplace=True)

df['Target'].head()

# %% [markdown]
# handle NaN rows

# %%
df = df.dropna()
print(df.head())

# %% [markdown]
# Seperate features and target 

# %%
feature_columns = [
    "Volatility",
    "Daily Return",
    "Volume Change",
    "five day average"
]
X = df[feature_columns]

Y = df['Target']

# %% [markdown]
# training and testing split 

# %%
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score,f1_score,roc_curve, roc_auc_score
# Make sure data is sorted by date
df = df.sort_index()

# choose split point
split_index = int(len(df) * 0.75)

# chronological split
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = Y.iloc[:split_index]
y_test = Y.iloc[split_index:]

# %% [markdown]
# choosing and training model
# 

# %%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {round(accuracy,2)}")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:" + str(cm))
precision = precision_score(y_test, y_pred)
print(f"Precision: {round(precision,2)}")
recall = recall_score(y_test, y_pred)
print(f"Recall: {round(recall,2)}")
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {round(f1,2)}")
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# %% [markdown]
# USING A KNN MODEL FOR VARIATION

# %%
# Import Library for KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# Create a training model KNeighborsClassifier
model_1 = KNeighborsClassifier()
model_1.fit(X_train, np.ravel(y_train))
# Model Evaluation
pred=model_1.predict(X_test)

# predict the class of the first 10 lines of the X_test dataset. The return will be an array containing the estimated categories.
y_test[0:10]
model_1.score(X_test, y_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, model_1.predict(X_test))
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
accuracy = accuracy_score(y_test, pred)
print(f"accuracy: {round(accuracy,2)}")

# %% [markdown]
# using a 3rd model and using a train test split according to date

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = df.sort_index()
split_index = int(len(df) * 0.75)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = Y.iloc[:split_index]
y_test = Y.iloc[split_index:]

rf = RandomForestRegressor(n_estimators=100, random_state=42)
y_pred = rf.fit(X_train, y_train).predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
single_data = X_test.iloc[0].values.reshape(1, -1)
single_pred = rf.predict(single_data)
print(f"Predicted value for the first test data point: {single_pred[0]:.2f}")
print(f"Actual value for the first test data point: {y_test.iloc[0]:.2f}")

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, model_1.predict(X_test))
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
accuracy = accuracy_score(y_test, pred)
print(f"accuracy: {round(accuracy,2)}")