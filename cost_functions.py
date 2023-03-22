"""
Implement cost functions/objectives here and use them to generate counterfactuals
"""
import numpy as np
import scipy.stats

def mad_weighted_distance(x_prime, x, X, weight_functions):
    mad = scipy.stats.median_abs_deviation(X, axis=0)
    # weight x and x_prime
    x_prime_weighted = np.array([weight_functions[i](x_prime[i]) for i in range(len(x_prime))])
    x_weighted = np.array([weight_functions[i](x[i]) for i in range(len(x))])
    distance = np.sum(np.abs(x_weighted-x_prime_weighted)/mad)
    return distance
    

def wachter2017_cost_function(x_prime: np.ndarray, x: np.ndarray, y_prime, lambda_value: float, model: object, X, weight_functions) -> float:
    distance = mad_weighted_distance(x_prime, x, X, weight_functions)
    prediction = model.predict_proba(x_prime.reshape((1, -1)))[0][0]
    misfit = (prediction-y_prime)**2
    return lambda_value * misfit + distance