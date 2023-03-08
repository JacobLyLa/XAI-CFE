import scipy.stats
import numpy as np
import pandas as pd
from cost_functions import wachter2017_cost_function
import optuna
# turn off optuna warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, x, y_prime_target, lambda_k, model, X, cost_function=wachter2017_cost_function, tol=0.05, framed=True):
    features = []
    ranges = [(0, 10), (100,200), (50, 200), (20, 50), (0, 100), (10, 100), (0, 5), (x[-1], 100)]
    # TODO: cleanup. last element is age. restrict to current_age - 100.
    is_int = [True, False, False, False, False, False, True, True]
    for i in range(len(ranges)):
        if not is_int[i]:
            features.append(trial.suggest_uniform(f'feature_{i}', ranges[i][0], ranges[i][1]))
        else:
            features.append(trial.suggest_int(f'feature_{i}', ranges[i][0], ranges[i][1]))

    # feature values to numpy array
    x_prime = []
    for i in range(len(features)):
        x_prime.append(features[i])
    x_prime = np.array(x_prime)

    return wachter2017_cost_function(x_prime, x, y_prime_target, lambda_k, model, X)





def get_counterfactuals(x, y_prime_target, model, X, cost_function=wachter2017_cost_function, tol=0.05, bnds=None, cons=None, framed=True):
    lambda_min =  1e-10
    lambda_max = 1e5
    lambda_steps = 30
    lambdas = np.logspace(np.log10(lambda_min), 
                            np.log10(lambda_max), 
                            lambda_steps)
    # scan over lambda
    candidates = []
    Y_primes = []
    for lambda_k in lambdas:
        print(f'lambda_k: {lambda_k}')
        arguments = x, y_prime_target, lambda_k, model, X
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, *arguments), n_trials=100)
        x_prime_hat = np.array(list(study.best_params.values()))
        print(f'cost value of best x\': {study.best_value}')
        # x_prime_hat = solution.x
        prediction = model.predict_proba(x_prime_hat.reshape((1, -1)))[0][0]
        Y_primes.append(prediction)
        candidates.append(x_prime_hat)
    Y_primes = np.array(Y_primes)
    candidates = np.array(candidates)
    # check if any counterfactual candidates meet the tolerance condition
    eps_condition = np.abs(Y_primes - y_prime_target) <= tol
    relevant_candidates =  candidates[eps_condition]
    if framed:
        return pd.DataFrame(data=relevant_candidates, columns=X.columns)
    return relevant_candidates