import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

# Create the input/output sequences for the models
X_train, y_train = create_sequences(train_data.values.reshape(-1, 1), seq_len)
X_test, y_test = create_sequences(test_data.values.reshape(-1, 1), seq_len)

# Define the random forest model
rf = RandomForestRegressor(n_estimators=100, max_depth=6)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Define the k-nearest neighbors model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)

# Define the support vector machines model
svm = SVR(kernel='rbf', C=1e3, gamma=0.1)
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)

# Define the Gaussian process regression model
gpr = GaussianProcessRegressor()
gpr.fit(X_train, y_train)
gpr_preds = gpr.predict(X_test)

# Evaluate the models using MAE and MSE
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_mse = mean_squared_error(y_test, rf_preds)
knn_mae = mean_absolute_error(y_test, knn_preds)
knn_mse = mean_squared_error(y_test, knn_preds)
svm_mae = mean_absolute_error(y_test, svm_preds)
svm_mse = mean_squared_error(y_test, svm_preds)
gpr_mae = mean_absolute_error(y_test, gpr_preds)
gpr_mse = mean_squared_error(y_test, gpr_preds)

# Print the evaluation results
print(f'Random Forest MAE: {rf_mae:.2f}')
print(f'Random Forest MSE: {rf_mse:.2f}')
print(f'K-Nearest Neighbors MAE: {knn_mae:.2f}')
print(f'K-Nearest Neighbors MSE: {knn_mse:.2f}')
print(f'Support Vector Machines MAE: {svm_mae:.2f}')
print(f'Support Vector Machines MSE: {svm_mse:.2f}')
print(f'Gaussian process regression MAE: {gpr_mae:.2f}')
print(f'Gaussian process regression MSE: {gpr_mse:.2f}')