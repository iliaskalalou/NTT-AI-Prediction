import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Series.__getitem__ treating keys as positions is deprecated.*")


def rolling_forecast_arima(series, train_size, order=(0, 1, 1)):
    """
    Performs a rolling forecast using an ARIMA model on a time series.

    The function splits the series into training and test sets, then iteratively 
    refits the ARIMA model on the history and makes a one-step-ahead forecast.
    The forecasts and actual test values are exponentiated to reverse any previous 
    logarithmic transformation.

    Parameters:
        series (array-like): The time series (e.g., log-transformed closing prices).
        train_size (int): The number of data points to use for training.
        order (tuple): The (p, d, q) parameters of the ARIMA model (default is (0, 1, 1)).

    Returns:
        tuple: (predictions, test_actual)
            - predictions: Array of forecasted values (exponentiated).
            - test_actual: Array of actual test values (exponentiated).
    """
    train, test = series[:train_size], series[train_size:]
    
    history = list(train)
    predictions = []
    
    for t in tqdm(range(len(test)), desc="Training ARIMA"):
        model = ARIMA(history, order=order)
        model_fit = model.fit(method_kwargs={'maxiter': 500})
        yhat = model_fit.forecast(steps=1)[0]
        predictions.append(yhat)
        history.append(test[t])
    
    predictions = np.exp(np.array(predictions))
    test_actual = np.exp(test)
    
    return predictions, test_actual
