import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
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

# Fit an ARCH model
arch_model = arch_model(train_data, vol='ARCH')
arch_fit = arch_model.fit()

# Make predictions on the test set using the ARCH model
arch_preds = arch_fit.forecast(horizon=len(test_data))

# Extract the predicted conditional variances from the ARCH model
arch_vars = arch_preds.variance.values[-1]

# Fit a GARCH model
garch_model = arch_model(train_data, vol='GARCH')
garch_fit = garch_model.fit()

# Make predictions on the test set using the GARCH model
garch_preds = garch_fit.forecast(horizon=len(test_data))

# Extract the predicted conditional variances from the GARCH model
garch_vars = garch_preds.variance.values[-1]

# Evaluate the models using MAE and MSE
arch_mae = mean_absolute_error(np.sqrt(arch_vars), np.sqrt(test_data))
arch_mse = mean_squared_error(np.sqrt(arch_vars), np.sqrt(test_data))
garch_mae = mean_absolute_error(np.sqrt(garch_vars), np.sqrt(test_data))
garch_mse = mean_squared_error(np.sqrt(garch_vars), np.sqrt(test_data))

# Print the evaluation results
print(f'ARCH MAE: {arch_mae:.2f}')
print(f'ARCH MSE: {arch_mse:.2f}')
print(f'GARCH MAE: {garch_mae:.2f}')
print(f'GARCH MSE: {garch_mse:.2f}')

# Plot the actual and predicted prices
plt.plot(test_data.index, np.sqrt(test_data), label='Actual')
plt.plot(test_data.index, np.sqrt(arch_vars), label='ARCH')
plt.plot(test_data.index, np.sqrt(garch_vars), label='GARCH')
plt.legend()
plt.show()
