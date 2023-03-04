import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import Input, Conv1D, Activation, Dropout, Lambda, Dense, LSTM, GRU, Attention, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import VGG16
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

# Define the input sequence length and number of features
seq_len = 30
n_features = 1

# Define a function to create the input/output sequences for the Seq2Seq model
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Create the input/output sequences for the Seq2Seq model
seq2seq_X_train, seq2seq_y_train = create_sequences(train_data.values.reshape(-1, 1), seq_len)
seq2seq_X_test, seq2seq_y_test = create_sequences(test_data.values.reshape(-1, 1), seq_len)

# Define the WaveNet model
def wavenet_model(seq_len, n_features):
    input_layer = Input(shape=(seq_len, n_features))
    residual = input_layer
    for i in range(6):
        dilation_rate = 2 ** i
        tanh_out = Conv1D(16, kernel_size=2, dilation_rate=dilation_rate,
                          padding='causal', name=f'block_{i}_tanh')(residual)
        sigm_out = Conv1D(16, kernel_size=2, dilation_rate=dilation_rate,
                          padding='causal', name=f'block_{i}_sigmoid')(residual)
        merge = Concatenate(name=f'block_{i}_merge')([tanh_out, sigm_out])
        skip_out = Conv1D(1, kernel_size=1, name=f'block_{i}_skip')(merge)
        residual = Lambda(lambda x: x[0] + x[1], name=f'block_{i}_residual')([residual, skip_out])
    output_layer = Activation('relu')(residual)
    output_layer = Conv1D(1, kernel_size=1, activation='linear', name='output_layer')(output_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define the Seq2Seq model
def seq2seq_model(seq_len, n_features):
    encoder_input = Input(shape=(seq_len, n_features))
    encoder = LSTM(64, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder(encoder_input)
    encoder_states = [state_h, state_c]

    decoder_input = Input(shape=(1, n_features))
    decoder_lstm = LSTM(64, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
    decoder_dense = Dense(1, activation='linear', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define the Attention Mechanisms model
def attention_model(seq_len, n_features):
    input_layer = Input(shape=(seq_len, n_features))
    lstm = LSTM(64, return_sequences=True)(input_layer)
    attention = Attention()([lstm, lstm])
    output_layer = Dense(1, activation='linear')(attention)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define the Transfer Learning model
def transfer_learning_model(seq_len, n_features):
    vgg = VGG16(include_top=False, input_shape=(seq_len, n_features, 3))
    for layer in vgg.layers:
        layer.trainable = False
    input_layer = Input(shape=(seq_len, n_features))
    reshape_layer = Lambda(lambda x: to_categorical(x))(input_layer)
    cnn_output = vgg(reshape_layer)
    cnn_output = Flatten()(cnn_output)
    dense_layer = Dense(64, activation='relu')(cnn_output)
    output_layer = Dense(1, activation='linear')(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define the RNN Attention model
def rnn_attention_model(seq_len, n_features):
    input_layer = Input(shape=(seq_len, n_features))
    lstm = LSTM(64, return_sequences=True)(input_layer)
    attention = Attention()([lstm, lstm])
    output_layer = Dense(1, activation='linear')(attention)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Fit the models and make predictions on the test set
wavenet = wavenet_model(seq_len, n_features)
wavenet.fit(train_data.values.reshape(-1, seq_len, n_features), train_data.values,
            epochs=100, batch_size=16, verbose=False,
            callbacks=[EarlyStopping(monitor='loss', patience=10)])
wavenet_preds = wavenet.predict(test_data.values.reshape(-1, seq_len, n_features))

seq2seq = seq2seq_model(seq_len, n_features)
seq2seq.fit([seq2seq_X_train, seq2seq_y_train], seq2seq_y_train,
            epochs=100, batch_size=16, verbose=False,
            callbacks=[EarlyStopping(monitor='loss', patience=10)])
seq2seq_preds = seq2seq.predict([seq2seq_X_test, np.zeros((len(seq2seq_X_test), 1, n_features))])

attention = attention_model(seq_len, n_features)
attention.fit(train_data.values.reshape(-1, seq_len, n_features), train_data.values,
              epochs=100, batch_size=16, verbose=False,              callbacks=[EarlyStopping(monitor='loss', patience=10)])
attention_preds = attention.predict(test_data.values.reshape(-1, seq_len, n_features))

transfer_learning = transfer_learning_model(seq_len, n_features)
transfer_learning.fit(pad_sequences([train_data.values], maxlen=seq_len, dtype='float32'),
                       train_data.values, epochs=100, batch_size=16, verbose=False,
                       callbacks=[EarlyStopping(monitor='loss', patience=10)])
transfer_learning_preds = transfer_learning.predict(pad_sequences([test_data.values], maxlen=seq_len, dtype='float32'))

rnn_attention = rnn_attention_model(seq_len, n_features)
rnn_attention.fit(train_data.values.reshape(-1, seq_len, n_features), train_data.values,
                   epochs=100, batch_size=16, verbose=False,
                   callbacks=[EarlyStopping(monitor='loss', patience=10)])
rnn_attention_preds = rnn_attention.predict(test_data.values.reshape(-1, seq_len, n_features))

# Evaluate the models using MAE and MSE
wavenet_mae = mean_absolute_error(test_data, wavenet_preds.flatten())
wavenet_mse = mean_squared_error(test_data, wavenet_preds.flatten())
seq2seq_mae = mean_absolute_error(seq2seq_y_test, seq2seq_preds.flatten())
seq2seq_mse = mean_squared_error(seq2seq_y_test, seq2seq_preds.flatten())
attention_mae = mean_absolute_error(test_data, attention_preds.flatten())
attention_mse = mean_squared_error(test_data, attention_preds.flatten())
transfer_learning_mae = mean_absolute_error(test_data[seq_len-1:], transfer_learning_preds.flatten())
transfer_learning_mse = mean_squared_error(test_data[seq_len-1:], transfer_learning_preds.flatten())
rnn_attention_mae = mean_absolute_error(test_data, rnn_attention_preds.flatten())
rnn_attention_mse = mean_squared_error(test_data, rnn_attention_preds.flatten())

# Print the evaluation results
print(f'WaveNet MAE: {wavenet_mae:.2f}')
print(f'WaveNet MSE: {wavenet_mse:.2f}')
print(f'Seq2Seq MAE: {seq2seq_mae:.2f}')
print(f'Seq2Seq MSE: {seq2seq_mse:.2f}')
print(f'Attention Mechanisms MAE: {attention_mae:.2f}')
print(f'Attention Mechanisms MSE: {attention_mse:.2f}')
print(f'Transfer Learning MAE: {transfer_learning_mae:.2f}')
print(f'Transfer Learning MSE: {transfer_learning_mse:.2f}')
print(f'RNN Attention MAE: {rnn_attention_mae:.2f}')
print(f'RNN Attention MSE: {rnn_attention_mse:.2f}')

# Plot the actual and predicted prices
plt.plot(test_data.index, test_data.values, label='Actual')
plt.plot(test_data.index[seq_len-1:], wavenet_preds.flatten(), label='WaveNet')
plt.plot(test_data.index[seq_len-1:], seq2seq_preds.flatten(), label='Seq2Seq')
plt.plot(test_data.index, attention_preds.flatten(), label='Attention Mechanisms')
plt.plot(test_data.index[seq_len-1:], transfer_learning_preds.flatten(), label='Transfer Learning')
plt.plot(test_data.index, rnn_attention_preds.flatten(), label='RNN Attention')
plt.legend()
plt.show()

