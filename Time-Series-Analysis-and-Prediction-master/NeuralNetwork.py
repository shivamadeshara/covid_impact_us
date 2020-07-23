import torch
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


file = pd.read_csv('/Users/vishalbansal/Downloads/us.csv')
file = file.drop('date',1)

num_days = 183

daily_cases = np.zeros(num_days)
cumulative_cases = np.zeros(num_days)

daily_deaths = np.zeros(num_days)
cumulative_deaths = np.zeros(num_days)

days = np.zeros(num_days)

for row in range(0,num_days):

    num_cases = (file.loc[row, 'cases'])
    num_deaths = (file.loc[row,'deaths'])

    cumulative_cases[row] = num_cases
    cumulative_deaths[row] = num_deaths
    if row == 0:
        daily_cases[row] = num_cases
        daily_deaths[row] = num_deaths
    else:
        daily_cases[row] = num_cases - cumulative_cases[row-1]
        daily_deaths[row] = num_deaths - cumulative_deaths[row - 1]


    days[row] = row

plt.plot(days,cumulative_cases)
plt.title("Cumulative cases")
plt.xlabel('days')
plt.ylabel('cases')
plt.show()

plt.plot(days,cumulative_deaths)
plt.title("Cumulative deaths")
plt.xlabel('days')
plt.ylabel('deaths')
plt.show()




data_range = (-1, 1)
scaler = MinMaxScaler(feature_range=data_range)        # scaler can also de-normalize the dataset by scaler.inverse_transform(), useful for actual prediction
dataset_scaled = scaler.fit_transform(dataset)
#dataset_scaled = numpy.array(dataset_scaled)


train_size = int(len(dataset_scaled) * 0.67)
test_size = len(dataset_scaled) - train_size
train, test = dataset_scaled[0:train_size,:], dataset_scaled[train_size:len(dataset),:]
print(len(train), len(test))


dataset = file.values
dataset = dataset.astype('float32')

def create_dataset(data, look_back=1):
    dataX, dataY = [], []
    i_range = len(data) - look_back - 1
    print(i_range)
    for i in range(0, i_range):
        dataX.append(data[i:(i + look_back)])  # index can move down to len(dataset)-1
        dataY.append(data[i + look_back])  # Y is the item that skips look_back number of items

    return np.array(dataX), np.array(dataY)


look_back = 4
dataX, dataY = create_dataset(dataset_scaled, look_back=look_back)

print("X shape:", dataX.shape)
print("Y shape:", dataY.shape)

print("Xt-3     Xt-2      Xt-1      Xt        Y")
print("---------------------------------------------")
for i in range(len(dataX)):
    print('%.2f   %.2f    %.2f    %.2f    %.2f' % (
    dataX[i][0][0], dataX[i][1][0], dataX[i][2][0], dataX[i][3][0], dataY[i][0]))