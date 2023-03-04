import yfinance as yf
import pandas as pd
import numpy as np
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
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

# Fit a Prophet model
prophet_model = Prophet()
prophet_data = pd.DataFrame({'ds': train_data.index, 'y': train_data.values})
prophet_fit = prophet_model.fit(prophet_data)
prophet_preds = prophet_fit.predict(pd.DataFrame({'ds': test_data.index}))

# Fit a feedforward neural network model
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_history = nn_model.fit(train_data.values.reshape(-1, 1), train_data.values,
                          epochs=100, batch_size=16, verbose=False, 
                          callbacks=[EarlyStopping(monitor='loss', patience=10)])
nn_preds = nn_model.predict(test_data.values.reshape(-1, 1))

# Fit an LSTM model
lstm_model = Sequential([
    LSTM(64, input_shape=(1, 1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_history = lstm_model.fit(train_data.values.reshape(-1, 1, 1), train_data.values, 
                              epochs=100, batch_size=16, verbose=False,
                              callbacks=[EarlyStopping(monitor='loss', patience=10)])
lstm_preds = lstm_model.predict(test_data.values.reshape(-1, 1, 1))

# Fit a GRU model
gru_model = Sequential([
    GRU(64, input_shape=(1, 1)),
    Dense(1)
])
gru_model.compile(optimizer='adam', loss='mean_squared_error')
gru_history = gru_model.fit(train_data.values.reshape(-1, 1, 1), train_data.values, 
                            epochs=100, batch_size=16, verbose=False,
                            callbacks=[EarlyStopping(monitor='loss', patience=10)])
gru_preds = gru_model.predict(test_data.values.reshape(-1, 1, 1))

# Fit a 1D convolutional neural network (CNN) model
cnn_model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(1, 1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(1)
])
cnn_model.compile(optimizer='adam', loss='mean_squared_error')
cnn_history = cnn_model.fit(train_data.values.reshape(-1, 1, 1), train_data.values,epochs=100, batch_size=16, verbose=False,
                            callbacks=[EarlyStopping(monitor='loss', patience=10)])
cnn_preds = cnn_model.predict(test_data.values.reshape(-1, 1, 1))

# Evaluate the models using MAE and MSE
prophet_mae = mean_absolute_error(test_data, prophet_preds['yhat'].values)
prophet_mse = mean_squared_error(test_data, prophet_preds['yhat'].values)
nn_mae = mean_absolute_error(test_data, nn_preds.flatten())
nn_mse = mean_squared_error(test_data, nn_preds.flatten())
lstm_mae = mean_absolute_error(test_data, lstm_preds.flatten())
lstm_mse = mean_squared_error(test_data, lstm_preds.flatten())
gru_mae = mean_absolute_error(test_data, gru_preds.flatten())
gru_mse = mean_squared_error(test_data, gru_preds.flatten())
cnn_mae = mean_absolute_error(test_data, cnn_preds.flatten())
cnn_mse = mean_squared_error(test_data, cnn_preds.flatten())

# Print the evaluation results
print(f'Prophet MAE: {prophet_mae:.2f}')
print(f'Prophet MSE: {prophet_mse:.2f}')
print(f'NN MAE: {nn_mae:.2f}')
print(f'NN MSE: {nn_mse:.2f}')
print(f'LSTM MAE: {lstm_mae:.2f}')
print(f'LSTM MSE: {lstm_mse:.2f}')
print(f'GRU MAE: {gru_mae:.2f}')
print(f'GRU MSE: {gru_mse:.2f}')
print(f'CNN MAE: {cnn_mae:.2f}')
print(f'CNN MSE: {cnn_mse:.2f}')

# Plot the actual and predicted prices
plt.plot(test_data.index, test_data.values, label='Actual')
plt.plot(test_data.index, prophet_preds['yhat'].values, label='Prophet')
plt.plot(test_data.index, nn_preds.flatten(), label='NN')
plt.plot(test_data.index, lstm_preds.flatten(), label='LSTM')
plt.plot(test_data.index, gru_preds.flatten(), label='GRU')
plt.plot(test_data.index, cnn_preds.flatten(), label='CNN')
plt.legend()
plt.show()

