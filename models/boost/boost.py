import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from matplotlib import pyplot as plt

# Define the stock symbol and date range
symbol = 'AAPL'
start_date = '2015-01-01'
end_date = '2021-12-31'

# Fetch the historical stock data using yfinance
stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)

# Extract the closing prices
prices = stock_data['Close']

# Split the data into training and testing sets
train_size = int(len(prices) * 0.8)
train_data, test_data = prices[:train_size], prices[train_size:]

# Define the input sequence length
seq_len = 30

# Define a function to create the input/output sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Create the input/output sequences for the XGBoost, LightGBM, and CatBoost models
X_train, y_train = create_sequences(train_data.values.reshape(-1, 1), seq_len)
X_test, y_test = create_sequences(test_data.values.reshape(-1, 1), seq_len)

# Define the XGBoost model
xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

# Define the LightGBM model
lgbm = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
lgbm.fit(X_train, y_train)
lgbm_preds = lgbm.predict(X_test)

# Define the CatBoost model
catboost = CatBoostRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
catboost.fit(X_train, y_train)
catboost_preds = catboost.predict(X_test)

# Evaluate the models using MAE and MSE
xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_mse = mean_squared_error(y_test, xgb_preds)
lgbm_mae = mean_absolute_error(y_test, lgbm_preds)
lgbm_mse = mean_squared_error(y_test, lgbm_preds)
catboost_mae = mean_absolute_error(y_test, catboost_preds)
catboost_mse = mean_squared_error(y_test, catboost_preds)

# Print the evaluation results
print(f'XGBoost MAE: {xgb_mae:.2f}')
print(f'XGBoost MSE: {xgb_mse:.2f}')
print(f'LightGBM MAE: {lgbm_mae:.2f}')
print(f'LightGBM MSE: {lgbm_mse:.2f}')
print(f'CatBoost MAE: {catboost_mae:.2f}')
print(f'CatBoost MSE: {catboost_mse:.2f}')

# Plot the actual and predicted prices
plt.plot(test_data.index[seq_len:], y_test)
plt.plot(test_data.index[seq_len:], y_test, label='Actual')
plt.plot(test_data.index[seq_len:], xgb_preds, label='XGBoost')
plt.plot(test_data.index[seq_len:], lgbm_preds, label='LightGBM')
plt.plot(test_data.index[seq_len:], catboost_preds, label='CatBoost')
plt.legend()
plt.show()

