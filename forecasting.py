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
# plt.show()

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

SARIMAXmodel = SARIMAX(y, order = (5, 4, 2), seasonal_order=(2,2,2,12))
SARIMAXmodel = SARIMAXmodel.fit()

months_to_predict=1
isYearFound = False
while not isYearFound:
    future_dates = pd.date_range("2024-01-01", periods=months_to_predict, freq='ME')
    future_df = pd.DataFrame(future_dates)
    future_df.index = pd.to_datetime(future_dates, format="%Y-%m-%d")
    y_pred = SARIMAXmodel.get_forecast(len(future_df))
    y_pred_df = y_pred.conf_int(alpha = 0.05) 
    y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
    y_pred_df.index = future_df.index
    y_pred_out = y_pred_df["Predictions"] 
    plt.plot(y_pred_out, color='Blue', label = 'Desktop Predictions')
    # plt.legend()
    
    for k, v in y_pred_out.items():
        if v >= 100:
            print("The year of Linux on the Desktop:", k)
            isYearFound = True
            break
    months_to_predict+=1

# plt.show()
plt.savefig("linux-growth.png")

