import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt

# Define the stock symbol and date range
symbol = 'AAPL'
start_date = '2015-01-01'
end_date = '2021-12-31'

# Fetch the historical stock data using yfinance
stock_data = yf.download(symbol, start=start_date, end=end_date)
print(stock_data)

# Extract the closing prices
prices = stock_data['Close']

# Split the data into training and testing sets
train_size = int(len(prices) * 0.8)
train_data, test_data = prices[:train_size], prices[train_size:]

# Fit an ARIMA model
arima_model = ARIMA(train_data, order=(1, 1, 1))
arima_fit = arima_model.fit()

# Make predictions on the test set using the ARIMA model
arima_preds = arima_fit.forecast(steps=len(test_data))[0]

# Fit a SARIMA model
sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit()

# Make predictions on the test set using the SARIMA model
sarima_preds = sarima_fit.forecast(steps=len(test_data))

# Evaluate the models using MAE and MSE
arima_mae = mean_absolute_error(test_data, arima_preds)
arima_mse = mean_squared_error(test_data, arima_preds)
sarima_mae = mean_absolute_error(test_data, sarima_preds)
sarima_mse = mean_squared_error(test_data, sarima_preds)

# Print the evaluation results
print(f'ARIMA MAE: {arima_mae:.2f}')
print(f'ARIMA MSE: {arima_mse:.2f}')
print(f'SARIMA MAE: {sarima_mae:.2f}')
print(f'SARIMA MSE: {sarima_mse:.2f}')

# Plot the actual and predicted prices
plt.plot(test_data.index, test_data.values, label='Actual')
plt.plot(test_data.index, arima_preds, label='ARIMA')
plt.plot(test_data.index, sarima_preds, label='SARIMA')
plt.legend()
plt.show()
