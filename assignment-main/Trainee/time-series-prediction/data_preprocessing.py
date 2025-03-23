import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jpholiday
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler

def convert_volume(x):
    """
    Converts volume strings from the '出来高' column into absolute numeric values.
    For example, "79.15M" becomes 79.15e6 and "1.03B" becomes 1.03e9.
    """
    if isinstance(x, str):
        if x.endswith('M'):
            return float(x.replace('M', '')) * 1e6
        elif x.endswith('B'):
            return float(x.replace('B', '')) * 1e9
        else:
            return float(x)
    return x


def load_and_preprocess_data(csv_file):
    """
    Loads data from a CSV file, converts the '日付け' column to datetime,
    cleans the '出来高' column by converting volume strings to numeric values,
    and reindexes the DataFrame to include all business days.
    """
    df = pd.read_csv(csv_file, encoding='utf-8')
    df['日付け'] = pd.to_datetime(df['日付け'])
    
    df['出来高'] = df['出来高'].apply(convert_volume)
    
    start_date = df['日付け'].min()
    end_date = df['日付け'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    df = df.set_index('日付け').reindex(all_dates)
    
    df = df.interpolate(method='linear', limit_direction='both')
    
    df = df.sort_index(ascending=False)
    return df



def normalize_data(df, features):
    """
    Normalizes the columns specified in 'features' using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[features] = scaler.fit_transform(df_norm[features])
    return df_norm, scaler



def create_sequences(data, seq_length):
    """
    Splits the time series 'data' into sequences of length 'seq_length'.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)


def visualisation(df):
    """
    Visualizes the data by plotting various graphs:
      - Trend of the closing price ('終値').
      - Descriptive statistics.
      - Distribution of the closing price.
      - Monthly variation of the closing price using a boxplot.
      - Performs an Augmented Dickey-Fuller test to check for stationarity.
    """
    # Set '日付け' as the index.
    df.set_index('日付け', inplace=True)

    # Plot the closing price trend.
    plt.figure(figsize=(12, 6))
    plt.plot(df['終値'], label='fence')
    plt.title("Trend in NTT closing price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    # Print descriptive statistics.
    print("Descriptive statistics:")
    print(df.describe())

    # Plot the distribution of the closing price.
    plt.figure(figsize=(12, 6))
    plt.title("Distribution of Closing Price")
    plt.xlabel("Closing Price")
    plt.ylabel("Frequency")
    sns.histplot(df['終値'], kde=True)
    plt.grid(True)

    # Add a 'Month' column and plot the monthly variation using a boxplot.
    df['Month'] = df.index.month
    plt.figure(figsize=(12, 6))
    plt.title("Variation of Closing Price by Month")
    plt.xlabel("Month")
    plt.ylabel("Closing Price")
    sns.boxplot(x='Month', y='終値', data=df)
    plt.grid(True)

    # Perform the Augmented Dickey-Fuller test.
    result = adfuller(df['終値'].dropna())
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])

    plt.show()



def information(df, matrix, start_date, end_date):
    """
    Displays basic information about the DataFrame and data matrix:
      - Prints the head of the DataFrame and the shape of the data matrix.
      - Counts the number of incomplete rows.
      - Calculates the number of holidays and missing dates.
    """
    print(df.head())
    print(matrix.shape)

    missing_rows_count = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == "":
                missing_rows_count += 1
                break

    print("Number of incomplete rows:", missing_rows_count)  # Expected to be 0 if all rows are complete

    holiday_count = 0
    for year in range(start_date.year, end_date.year + 1):
        for date_str, holiday_name in jpholiday.year_holidays(year):
            holiday_date = pd.to_datetime(date_str)
            if start_date <= holiday_date <= end_date:
                if holiday_date.weekday() < 5:  # Monday=0, Friday=4
                    holiday_count += 1
    print("Number of holidays in all years:", holiday_count)

    df['日付け'] = pd.to_datetime(df['日付け'])  # Ensure the date column is datetime
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    missing_dates = all_dates.difference(df['日付け'])
    print("Missing dates:", missing_dates)
    print("Percentage of missing dates:", ((len(missing_dates) - holiday_count) / len(all_dates)) * 100, "% LOST")



def plot_forecast(train_log, test_actual, forecast, test_index,
                  title, xlabel, ylabel, train_label, real_label, forecast_label):
    """
    Plots forecast results by comparing training data, actual test values, and forecasted values.
    
    Parameters:
      - train_log: Training data in log scale (if applicable).
      - test_actual: Actual test data (after applying the inverse transformation if needed).
      - forecast: Forecasted values.
      - test_index: Index for test data used for plotting.
      - title: Title of the plot.
      - xlabel: Label for the x-axis.
      - ylabel: Label for the y-axis.
      - train_label: Label for training data in the legend.
      - real_label: Label for actual test data in the legend.
      - forecast_label: Label for forecasted data in the legend.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(np.exp(train_log), label=train_label)
    plt.plot(test_actual, label=real_label)
    plt.plot(pd.Series(forecast, index=test_index), label=forecast_label, linestyle="--")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()




def adjust_monthly_volatility(df, column='終値', reduction_percentage=0.3, threshold_multiplier=1.3):
    """
    Adjusts the volatility of the specified column on a monthly basis.
    For each month, if a value exceeds threshold_multiplier times the monthly average,
    it reduces the deviation from the mean by reduction_percentage (e.g., 0.3 for 30% reduction).
    
    Parameters:
      - df: DataFrame with a datetime index.
      - column: The column to adjust (default is '終値').
      - reduction_percentage: The percentage by which to reduce the deviation (default 0.3).
      - threshold_multiplier: The multiplier threshold relative to the monthly mean (default 1.3).
    
    Returns:
      The adjusted DataFrame.
    """
    df_adjusted = df.copy()

    if not isinstance(df_adjusted.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be of type DatetimeIndex.")
    
    df_adjusted['Month'] = df_adjusted.index.to_period('M')
    
    monthly_means = df_adjusted.groupby('Month')[column].mean()
    
    def adjust_value(row):
        month = row['Month']
        mean_val = monthly_means[month]

        if row[column] > mean_val * threshold_multiplier:
            return mean_val + (row[column] - mean_val) * (1 - reduction_percentage)
        else:
            return row[column]
    
    df_adjusted[column] = df_adjusted.apply(adjust_value, axis=1)

    df_adjusted.drop(columns='Month', inplace=True)
    return df_adjusted

