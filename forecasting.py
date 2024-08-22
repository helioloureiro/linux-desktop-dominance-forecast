#! /usr/bin/env python3

## Data source: https://gs.statcounter.com/os-market-share/desktop/worldwide/chart.php?device=Desktop&device_hidden=desktop&statType_hidden=os_combined&region_hidden=ww&granularity=monthly&statType=Operating%20System&region=Worldwide&fromInt=200901&toInt=202407&fromMonthYear=2009-01&toMonthYear=2024-07&csv=1
## Destination: os_combined-ww-monthly-200901-202407.csv 

## Code based on: https://github.com/spierre91/builtiin/blob/main/time_series_forecasting.py
## Article: https://builtin.com/data-science/time-series-forecasting-python

import pandas as pd 
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

dt = pd.read_csv("os_combined-ww-monthly-200901-202407.csv")


dt.index = pd.to_datetime(dt['Date'], format='%Y-%m')
del dt['Date']

print(dt.head())
sns.set()
plt.ylabel('Percentage of desktops')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.plot(dt.index, dt['Linux'], )
plt.show()

train = dt[dt.index < pd.to_datetime("2024-01-01", format='%Y-%m-%d')]
test = dt[dt.index >= pd.to_datetime("2024-01-01", format='%Y-%m-%d')]
print(test)
plt.plot(train, color = "black", label = 'Training')
plt.plot(test, color = "red", label = 'Testing')
plt.ylabel('Percentage of desktops')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for Desktop domination")

y = train['Linux']

ARMAmodel = SARIMAX(y, order = (1, 0, 1))
ARMAmodel = ARMAmodel.fit()

y_pred = ARMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 
plt.plot(y_pred_out, color='green', label = 'ARMA Predictions')
plt.legend()


import numpy as np
from sklearn.metrics import mean_squared_error

arma_rmse = np.sqrt(mean_squared_error(test["Linux"].values, y_pred_df["Predictions"]))
print("ARMA RMSE: ",arma_rmse)




ARIMAmodel = ARIMA(y, order = (5, 4, 2))
ARIMAmodel = ARIMAmodel.fit()

y_pred = ARIMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 
plt.plot(y_pred_out, color='Yellow', label = 'ARIMA Predictions')
plt.legend()


import numpy as np
from sklearn.metrics import mean_squared_error

arma_rmse = np.sqrt(mean_squared_error(test["Linux"].values, y_pred_df["Predictions"]))
print("ARIMA RMSE: ",arma_rmse)



SARIMAXmodel = SARIMAX(y, order = (5, 4, 2), seasonal_order=(2,2,2,12))
SARIMAXmodel = SARIMAXmodel.fit()

y_pred = SARIMAXmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 
plt.plot(y_pred_out, color='Blue', label = 'SARIMA Predictions')
plt.legend()


import numpy as np
from sklearn.metrics import mean_squared_error

arma_rmse = np.sqrt(mean_squared_error(test["Linux"].values, y_pred_df["Predictions"]))
print("SARIMA RMSE: ",arma_rmse)

plt.show()
plt.savefig("linux-growth.png")

