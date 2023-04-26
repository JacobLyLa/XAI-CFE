"""
Implement cost functions/objectives here and use them to generate counterfactuals
"""

import numpy as np
import pandas as pd

def create_weighted_wachter2017_cost_function(X: pd.DataFrame, x: pd.DataFrame, y_target: float, model: object, features: list):
    x.reset_index(drop=True, inplace=True)
    def apply_feature_weights(X, features):
        X_weighted = X.copy()
        for feature in features:
            X_weighted[feature.name] = X[feature.name].apply(feature.weight_function)
        return X_weighted
    
    cat_idx = X.select_dtypes(include=['object', 'bool']).columns
    num_idx = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Compute normalized data
    medians = X.median()
    mad = (X - medians).abs().median()
    epsilon = 1e-4
    mad[mad == 0] = epsilon
    x_normalized = (x - medians) / mad
    
    # Apply feature weights to x
    apply_feature_weights(x, features)
    
    # Put categorical features back
    x_normalized[cat_idx] = x[cat_idx]

    # Define Wachter2017 cost function
    def weighted_wachter2017_cost_function(x_prime: pd.DataFrame, lambda_value: float) -> float:
        x_prime.reset_index(drop=True, inplace=True)
        # Normalize x_prime
        x_prime_normalized = (x_prime - medians) / mad
        # Put categorical features back
        x_prime_normalized[cat_idx] = x_prime[cat_idx]
        # apply feature weights to x_prime
        x_prime_normalized = apply_feature_weights(x_prime_normalized, features)
        
        # Compute distances
        numeric_distance = np.sum(np.minimum((x_normalized[num_idx] - x_prime_normalized[num_idx])**2, 1).sum())
        categorical_distance = x_prime_normalized[cat_idx].sum(axis=1).sum()
        
        # Compute misfit
        prediction = model.predict_proba(x_prime)[0][0]
        misfit = (prediction - y_target)**2
        
        return lambda_value * misfit + numeric_distance + categorical_distance
    
    return weighted_wachter2017_cost_function

def create_wachter2017_cost_function(X: pd.DataFrame, x: pd.DataFrame, y_target: float, model: object):
    x.reset_index(drop=True, inplace=True)
    cat_idx = X.select_dtypes(include=['object', 'bool']).columns
    num_idx = X.select_dtypes(include=['int64', 'float64']).columns
    # Compute normalized data
    medians = X.median()
    mad = (X - medians).abs().median()
    epsilon = 1e-4
    mad[mad == 0] = epsilon
    x_normalized = (x - medians) / mad
    
    # Put categorical features back
    x_normalized[cat_idx] = x[cat_idx]

    # Define Wachter2017 cost function
    def wachter2017_cost_function(x_prime: pd.DataFrame, lambda_value: float) -> float:
        x_prime.reset_index(drop=True, inplace=True)
        # Normalize x_prime
        x_prime_normalized = (x_prime - medians) / mad
        
        # Put categorical features back
        x_prime_normalized[cat_idx] = x_prime[cat_idx]
        
        # Compute distances
        numeric_distance = np.sum(np.minimum((x_normalized[num_idx] - x_prime_normalized[num_idx])**2, 1).sum())
        categorical_distance = np.sum(x_normalized[cat_idx] != x_prime_normalized[cat_idx]).sum()
        
        # Compute misfit
        prediction = model.predict_proba(x_prime)[0][0]
        misfit = (prediction - y_target)**2
        
        return lambda_value * misfit + numeric_distance + categorical_distance
    
    return wachter2017_cost_function