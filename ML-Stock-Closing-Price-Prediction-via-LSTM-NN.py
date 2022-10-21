# import all relevant libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt

# set the number of days variable
days = 7

# implement the pandas datareader source along with stock choice and range of dataset
data = web.DataReader('TSLA', 'stooq', start='2012-01-01', end='2020-01-29').iloc[::-1]

# display your chosen financial data as a Pandas DataFrame
data

# isolate 'closing price' column of stock data
filtered_data = data.filter(['Close'])
close_prices = filtered_data.values

# set 80% of the closing price data to be used for training the model
train_length_test = len(close_prices) * 0.8

# ctrl-F or HASH to see train_length impact on code
# train_length = 1534
close_prices.shape

# return length of training data to verify that it is 80% of initial dataset
train_length_test

# pre-process data by scaling financial values between 0 and 1 to improve model accuracy
myScaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = myScaler.fit_transform(close_prices)
train_data = scaled_data[0:train_length, :]

# initialise empty x and y training sets
train_x = []
train_y = []

# create for loop: x will contain data from 0 -> (days-1), y will contain singular value of (days+1)
for i in range(days, len(train_data)):
    train_x.append(train_data[i - days:i, 0])
    train_y.append(train_data[i, 0])

# convert to numpy arrays to enable model to undergo matrix operations
    train_x = np.array(train_x)
train_y = np.array(train_y)
train_x.shape

# give x-training set 3 dimensions
train_x = np.reshape(train_x, (train_x.shape[0], days, 1))

# define ML and NN model architecture
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(days, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(25))
model.add(Dense(1))

# define omptimiser and loss functions, select batch size and epochs
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_x, train_y, batch_size=1, epochs=10)

# isolate testing portion of data, create test sets, append past 'days' values of data to our x-test set
test_data = scaled_data[train_length - days:, :]
x_test = []
y_test = close_prices[train_length:, :]
for i in range(days, len(test_data)):
    x_test.append(test_data[i - days:i, 0])

# convert x-test set into numpy array
x_test = np.array(x_test)
x_test.shape

# reshape x-testing set into a 3D array, provide inverse scaling to un-scale the financial data back to relevant values, show model rmse
x_test = np.reshape(x_test, (x_test.shape[0], days, 1))
predictions = model.predict(x_test)
predictions = myScaler.inverse_transform(predictions)
rmse = np.sqrt(((predictions - y_test) ** 2).mean())
print(rmse)

# implement a validation set from index 0 to training data length, give a column to the 'predictions' variable
train = filtered_data[:train_length]
valid = filtered_data[train_length:]
valid['Predictions'] = predictions

# visualise model performance
plt.figure(figsize=(16, 8))
plt.title("LSTM Model")
plt.plot(train["Close"])
plt.plot(valid[['Close', 'Predictions']])
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend(['Training Data', 'True Closing Price', 'Predictions'])

# BONUS: predict the stock closing price for a time in which no data is given (e.g. stooq data end date + 1 day)
last_num_days = close_prices[-1 * days:]
last_num_days_scaled = myScaler.transform(last_num_days)
X_test = []
X_test.append(last_num_days_scaled)
X_test = np.array(X_test)
pred_price = model.predict(X_test)
pred_price = myScaler.inverse_transform(pred_price)
print(pred_price)
