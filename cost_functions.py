"""
Implement cost functions/objectives here and use them to generate counterfactuals
"""
import numpy as np
import scipy.stats

#TODO: specify types
def wachter2017_cost_function(x_prime: np.ndarray, x: np.ndarray, y_prime, lambda_value: float, model: object, X) -> float:
    mad =  scipy.stats.median_abs_deviation(X, axis=0)
    distance = np.sum(np.abs(x-x_prime)/mad)
    prediction = model.predict_proba(x_prime.reshape((1, -1)))[0][0]
    misfit = (prediction-y_prime)**2
    return lambda_value * misfit + distance