import numpy as np
import optuna
import pandas as pd

# turn off optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

# if very small float changes, revert to original value
def epsilon_rounding(x, x_prime, epsilon):
    for feature in x.columns.values:
        if type(x[feature].values[0]) == np.float64:
            if np.abs(x[feature].values[0] - x_prime[feature].values[0]) <= epsilon:
                x_prime[feature] = x[feature].values[0]

def get_counterfactuals(x, y_target, model, cost_function, features, tol, optimization_steps):    
    lambda_min, lambda_max, lambda_steps = 1e-2, 1e2, 5
    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), lambda_steps)
    candidates, y_primes = [], []
 
    def objective(trial, x, lambda_k, cost_function, features):
        x_prime = x.copy()
        for feature in features:
            feature.sample(trial)
            x_prime[feature.name] = feature.value
        epsilon_rounding(x, x_prime, 1e-3)
        return cost_function(x_prime, lambda_k)

    # scan over lambda
    for lambda_k in lambdas:
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda trial: objective(trial, x, lambda_k, cost_function, features), n_trials=optimization_steps)
  
        # get best counterfactual candidate
        x_prime = x.copy()
        for param in x.columns.values:
            if param in study.best_params:
                x_prime[param] = study.best_params[param]
        epsilon_rounding(x, x_prime, 1e-3)
        # predict DataFrame
        prediction = model.predict_proba(x_prime)[0][0]
        y_primes.append(prediction)
        candidates.append(x_prime.values[0])
    y_primes = np.array(y_primes)
    candidates = np.array(candidates)
    # check if any counterfactual candidates meet the tolerance condition
    eps_condition = np.abs(y_primes - y_target) <= tol
    relevant_candidates = candidates[eps_condition]
    # to dataframe, use cols from x
    relevant_candidates = pd.DataFrame(relevant_candidates, columns=x.columns)
    # remove duplicates and reset index
    relevant_candidates = relevant_candidates.drop_duplicates().reset_index(drop=True)
    return relevant_candidates