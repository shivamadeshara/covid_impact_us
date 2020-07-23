import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

file = pd.read_csv('/Users/vishalbansal/Downloads/us.csv')
file = file.iloc[:]

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


# Below not done yet
# test_data_size1 = 140
# train_data1 = daily_cases[:-test_data_size1]
# test_data1 = daily_cases[-test_data_size1:]
#
# test_data_size2 = 140
# train_data2 = daily_deaths[:-test_data_size2]
# test_data2 = daily_deaths[-test_data_size2:]


