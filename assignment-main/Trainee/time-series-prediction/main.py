import matplotlib.pyplot as plt
import pandas as pd
import jpholiday
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.stattools import adfuller
from sklearn.impute import SimpleImputer
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import data_preprocessing as preprocessing
import model_arima as model_arima
import model_gru as model_gru
import model_lstm as model_lstm
import model_tcn as model_tcn
import evaluation as evaluation


warnings.filterwarnings("ignore", category=FutureWarning, message=".*FutureWarning:.*")
warnings.filterwarnings("ignore", message=".*DataFrame.interpolate with object dtype is deprecated.*")


#日付け   Date  
#終値  Close 
#始値    Open   
#高値     Hight
#安値     Low
#出来高    Volume
#変化率 %   Change



# ============================ INITIALIZATION ============================

# Read CSV data and convert '日付け' (date) column to datetime.

df = pd.read_csv('stock_price.csv', encoding='utf-8')

df['日付け'] = pd.to_datetime(df['日付け'])

start_date = df['日付け'].min()

end_date = df['日付け'].max()

matrix = df.values


print(df.head())
print(matrix.shape)

missing_rows_count = 0
for i in range(len(matrix)):
    for j in range(len(matrix[0])):
        if matrix[i][j] == "":
            missing_rows_count += 1
            break

#print("Number of incomplete rows:", missing_rows_count)  # Expected to be 0 if all rows are complete

holiday_count = 0
for year in range(start_date.year, end_date.year + 1):
    for date_str, holiday_name in jpholiday.year_holidays(year):
        holiday_date = pd.to_datetime(date_str)
        if start_date <= holiday_date <= end_date:
            if holiday_date.weekday() < 5:  # Monday=0, Friday=4
                holiday_count += 1
#print("Number of holidays in all years:", holiday_count)

df['日付け'] = pd.to_datetime(df['日付け'])  # Ensure the date column is datetime
all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
missing_dates = all_dates.difference(df['日付け'])

#print("Missing dates:", missing_dates)
print("Percentage of missing dates:", ((len(missing_dates) - holiday_count) / len(all_dates)) * 100, "% LOST")


#preprocessing.visualisation(df)


# ======================== DATA PRE-PROCESSING ===========================

df_copy = df.copy()
df_copy.set_index('日付け', inplace=True)
df_copy.sort_index(inplace=True)

# Create a complete index of business days between the first and last date.
complete_index = pd.date_range(start=df_copy.index.min(), end=df_copy.index.max(), freq='B')
df_complete = df_copy.reindex(complete_index)
df_complete = df_complete.infer_objects(copy=False)

df_complete = df_complete.interpolate(method='linear', limit_direction='both')
df_complete = df_complete.sort_index(ascending=True)
print("New DataFrame shape:", df_complete.shape)


features = ['終値', '始値', '高値', '安値', '出来高']
scaler = MinMaxScaler()
df_normalized = df_complete.copy()
df_normalized['出来高'] = df_normalized['出来高'].apply(preprocessing.convert_volume)
df_complete['出来高'] = df_complete['出来高'].apply(preprocessing.convert_volume)
df_normalized, scaler = preprocessing.normalize_data(df_normalized, features)


# ============================== ARIMA ==============================

# Working with non-normalized data first.

df_arima = df_complete.copy().sort_index()

df_arima['終値_log'] = np.log(df_arima['終値'] + 1e-6)

# useless
#df_arima = preprocessing.adjust_monthly_volatility(df_arima, column='終値', reduction_percentage=0.3, threshold_multiplier=1.3)

print("\n=== ARIMA (Non Normalized) ===")

closing_prices_log = df_arima['終値_log']


result = adfuller(closing_prices_log.dropna())

print("ADF Statistic (log-transformed):", result[0])
print("p-value (log-transformed):", result[1])


train_size = int(len(closing_prices_log) * 0.95)
train_log = closing_prices_log[:train_size]
test_log  = closing_prices_log[train_size:]
print("Training size (log):", len(train_log))
print("Test size (log):", len(test_log))


predictions_log, test_actual = model_arima.rolling_forecast_arima(closing_prices_log, train_size, order=(0, 1, 1))

forecast = predictions_log

rmse = np.sqrt(mean_squared_error(test_actual, forecast))
mae  = mean_absolute_error(test_actual, forecast)
print("RMSE:", rmse)
print("MAE :", mae)

direction_accuracy, proximity_accuracy, global_score = evaluation.evaluate_forecast(test_actual, forecast)
print("ARIMA Direction Accuracy (%):", direction_accuracy)
#print("ARIMA Proximity Accuracy (%):", proximity_accuracy)



plt.figure(figsize=(12, 6))
plt.plot(test_actual, label="Actual")
plt.plot(pd.Series(forecast, index=test_log.index), label="ARIMA Forecast", linestyle="--")
plt.title("ARIMA Forecast vs Actual Values")
plt.xlabel("Time Steps")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
#plt.show()



print("\n=== ARIMA (Normalized) ===")

# Now using normalized data.

df_arima = df_normalized.copy().sort_index()

df_arima['終値_log'] = np.log(df_arima['終値'] + 1e-6)

closing_prices_log = df_arima['終値_log']


result = adfuller(closing_prices_log.dropna())

print("ADF Statistic (log-transformed) [Normalized]:", result[0])
print("p-value (log-transformed) [Normalized]:", result[1])

predictions_log, test_actual = model_arima.rolling_forecast_arima(closing_prices_log, train_size, order=(0, 1, 1))

forecast = predictions_log

rmse = np.sqrt(mean_squared_error(test_actual, forecast))
mae  = mean_absolute_error(test_actual, forecast)
print("ARIMA RMSE [Normalized]:", rmse)
print("ARIMA MAE [Normalized]:", mae)

direction_accuracy, proximity_accuracy, global_score = evaluation.evaluate_forecast(test_actual, forecast)
print("ARIMA Direction Accuracy (%)[Normalized]:", direction_accuracy)
#print("ARIMA Proximity Accuracy (%)[Normalized]:", proximity_accuracy)


plt.figure(figsize=(12, 6))
plt.plot(np.exp(train_log), label="Training")
plt.plot(test_actual, label="Actual")
plt.plot(pd.Series(forecast, index=test_log.index), label="ARIMA Forecast", linestyle="--")
plt.title("ARIMA Forecast vs Actual Values (Normalized)")
plt.xlabel("Time Steps")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
#plt.show()


# ============================== LSTM ==============================

seq_length = 10
train_ratio = 0.95
num_epochs = 50

print("\n=== LSTM (Non Normalized) ===")
series_non_norm = df_complete['終値'].values  
y_test_non_norm, predictions_non_norm, rmse_non_norm, mae_non_norm = model_lstm.train_and_evaluate(
    series_non_norm, seq_length, train_ratio, num_epochs
)
print("LSTM (Non Normalized) RMSE:", rmse_non_norm)
print("LSTM (Non Normalized) MAE:", mae_non_norm)

direction_accuracy, proximity_accuracy, global_score = evaluation.evaluate_forecast(y_test_non_norm, predictions_non_norm)
print("LSTM (Non Normalized) Direction Accuracy (%):", direction_accuracy)
#print("LSTM (Non Normalized) Proximity Accuracy (%):", proximity_accuracy)
#print("LSTM (Non Normalized) Global Score (%):", global_score)

plt.figure(figsize=(12, 6))
plt.plot(y_test_non_norm, label="Actual")
plt.plot(predictions_non_norm, label="LSTM Predictions (Non Normalized)", linestyle="--")
plt.title("LSTM Forecast vs Actual Values (Non Normalized)")
plt.xlabel("Time Steps")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
#plt.show()


print("\n=== LSTM (Normalized) ===")
series_norm = df_normalized['終値'].values  
y_test_norm, predictions_norm, rmse_norm, mae_norm = model_lstm.train_and_evaluate(
    series_norm, seq_length, train_ratio, num_epochs
)
print("LSTM (Normalized) RMSE:", rmse_norm)
print("LSTM (Normalized) MAE:", mae_norm)

direction_accuracy, proximity_accuracy, global_score = evaluation.evaluate_forecast(y_test_norm, predictions_norm)
print("LSTM (Normalized) Direction Accuracy (%):", direction_accuracy)
#print("LSTM (Normalized) Proximity Accuracy (%):", proximity_accuracy)
#print("LSTM (Normalized) Global Score (%):", global_score)

plt.figure(figsize=(12, 6))
plt.plot(y_test_norm, label="Actual")
plt.plot(predictions_norm, label="LSTM Predictions (Normalized)", linestyle="--")
plt.title("LSTM Forecast vs Actual Values (Normalized)")
plt.xlabel("Time Steps")
plt.ylabel("Normalized Closing Price")
plt.legend()
plt.grid(True)
#plt.show()



# ============================== GRU ==============================

seq_length = 10
train_ratio = 0.95
num_epochs = 50

print("\n=== GRU (Non Normalized) ===")
series_norm = df_complete['終値'].values  
y_test_gru, predictions_gru, rmse_gru, mae_gru = model_gru.train_and_evaluate_gru(
    series_norm, seq_length, train_ratio, num_epochs
)
print("GRU (Non Normalized) RMSE:", rmse_gru)
print("GRU (Non Normalized) MAE:", mae_gru)

direction_accuracy, proximity_accuracy, global_score = evaluation.evaluate_forecast(y_test_gru, predictions_gru)
print("GRU (Non Normalized) Direction Accuracy (%):", direction_accuracy)
#print("GRU (Non Normalized) Proximity Accuracy (%):", proximity_accuracy)
#print("GRU (Non Normalized) Global Score (%):", global_score)

plt.figure(figsize=(12, 6))
plt.plot(y_test_gru, label="Actual")
plt.plot(predictions_gru, label="GRU Predictions", linestyle="--")
plt.title("GRU Forecast vs Actual Values")
plt.xlabel("Time Steps")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
#plt.show()



print("\n=== GRU (Normalized) ===")

seq_length = 10
train_ratio = 0.95
num_epochs = 50

series_norm = df_normalized['終値'].values

y_test_gru, predictions_gru, rmse_gru, mae_gru = model_gru.train_and_evaluate_gru(
    series_norm, seq_length, train_ratio, num_epochs
)

print("GRU (Normalized) RMSE:", rmse_gru)
print("GRU (Normalized) MAE:", mae_gru)


direction_accuracy, proximity_accuracy, global_score = evaluation.evaluate_forecast(y_test_gru, predictions_gru)
print("GRU (Normalized) Direction Accuracy (%):", direction_accuracy)
#print("GRU (Normalized) Proximity Accuracy (%):", proximity_accuracy)
#print("GRU (Normalized) Global Score (%):", global_score)


plt.figure(figsize=(12, 6))
plt.plot(y_test_gru, label="Real")
plt.plot(predictions_gru, label="GRU Predictions", linestyle="--")
plt.title("GRU Forecast vs Real Values (Normalized)")
plt.xlabel("Time Steps")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
#plt.show()


# ============================== TCN ==============================

seq_length = 10
train_ratio = 0.95
num_epochs_tcn = 50

print("\n=== TCN (Non Normalized) ===")
series_norm = df_complete['終値'].values  
y_test_tcn, predictions_tcn, rmse_tcn, mae_tcn = model_tcn.train_and_evaluate_tcn(
    series_norm, seq_length, train_ratio, num_epochs_tcn, lr=0.001
)
print("TCN (Non Normalized) RMSE:", rmse_tcn)
print("TCN (Non Normalized) MAE:", mae_tcn)

direction_accuracy, proximity_accuracy, global_score = evaluation.evaluate_forecast(y_test_tcn, predictions_tcn)
print("TCN (Non Normalized) Direction Accuracy (%):", direction_accuracy)
#print("TCN (Non Normalized) Proximity Accuracy (%):", proximity_accuracy)
#print("TCN (Non Normalized) Global Score (%):", global_score)

plt.figure(figsize=(12, 6))
plt.plot(y_test_tcn, label="Actual")
plt.plot(predictions_tcn, label="TCN Predictions", linestyle="--")
plt.title("TCN Forecast vs Actual Values (Non Normalized)")
plt.xlabel("Time Steps")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
#plt.show()


print("\n=== TCN (Normalized)  ===")

series_norm = df_normalized['終値'].values

y_test_tcn, predictions_tcn, rmse_tcn, mae_tcn = model_tcn.train_and_evaluate_tcn(
    series_norm, seq_length, train_ratio, num_epochs_tcn, lr=0.001
)

# Affichage des métriques classiques
print("TCN (Normalized)  RMSE:", rmse_tcn)
print("TCN (Normalized)  MAE:", mae_tcn)

# Évaluation de la fiabilité du modèle TCN
direction_accuracy, proximity_accuracy, global_score = evaluation.evaluate_forecast(y_test_tcn, predictions_tcn)
print("TCN (Normalized) Précision de la tendance (%):", direction_accuracy)
#print("TCN (Normalized) Précision de proximité moyenne (%):", proximity_accuracy)
#print("TCN (Normalized) Score global de fiabilité (%):", global_score)


plt.figure(figsize=(12, 6))
plt.plot(y_test_tcn, label="Real")
plt.plot(predictions_tcn, label="TCN Predictions", linestyle="--")
plt.title("TCN Forecast vs Real Values (Normalized) ")
plt.xlabel("Time Steps")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
#plt.show()





