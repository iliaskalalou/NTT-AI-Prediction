import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_forecast(actual, forecast, threshold=1.0):
    """
    Computes evaluation metrics for a forecast by comparing the daily changes 
    between actual and forecasted values.

    The function calculates:
      - Direction accuracy: the percentage of days where the forecast correctly 
        predicts the direction (up or down) of the change.
      - Proximity accuracy: a score based on the absolute difference between the 
        forecast and the actual value (100% if difference is 0, 0% if difference 
        is greater than or equal to the threshold).
      - Global score: the average of direction and proximity accuracies.

    Parameters:
        actual (array-like): Array of actual values.
        forecast (array-like): Array of forecasted values.
        threshold (float): Threshold for proximity evaluation (default is 1.0).

    Returns:
        tuple: (direction_accuracy, proximity_accuracy, global_score)
    """
    direction_correct = 0
    proximity_scores = []
    n = len(actual)
    
    for i in range(1, n):
        actual_change = actual[i] - actual[i - 1]
        forecast_change = forecast[i] - forecast[i - 1]
        if (actual_change >= 0 and forecast_change >= 0) or (actual_change < 0 and forecast_change < 0):
            direction_correct += 1
        diff = abs(forecast[i] - actual[i])
        score = 0 if diff >= threshold else (1 - diff/threshold) * 100
        proximity_scores.append(score)
    
    direction_accuracy = (direction_correct / (n - 1)) * 100
    proximity_accuracy = np.mean(proximity_scores)
    global_score = (direction_accuracy + proximity_accuracy) / 2
    
    return direction_accuracy, proximity_accuracy, global_score
