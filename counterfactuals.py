import scipy.stats
import numpy as np
import pandas as pd
from cost_functions import wachter2017_cost_function
import optuna
# turn off optuna warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, x, y_prime_target, lambda_k, model, X, cost_function, boundaries):
    features = []
    i = 0
    for feature in boundaries:
        if feature[0] is int:
            features.append(trial.suggest_int(f'int_feature_{i}', feature[1], feature[2]))
        elif feature[0] is float:
            features.append(trial.suggest_uniform(f'float_feature_{i}', feature[1], feature[2]))
        i += 1

    # feature values to numpy array
    x_prime = []
    for i in range(len(features)):
        x_prime.append(features[i])
    x_prime = np.array(x_prime)

    return cost_function(x_prime, x, y_prime_target, lambda_k, model, X)

def get_counterfactuals(x, y_prime_target, model, X, cost_function, boundaries, tol=0.05, optimization_method="optuna", optimization_steps=10, framed=True):    
    lambda_min =  1e-10
    lambda_max = 1e5
    lambda_steps = 10
    lambdas = np.logspace(np.log10(lambda_min), 
                            np.log10(lambda_max), 
                            lambda_steps)
    candidates = []
    Y_primes = []
    # scan over lambda

    if (optimization_method=="optuna"):
        for lambda_k in lambdas:
            objective_arguments = x, y_prime_target, lambda_k, model, X, cost_function, boundaries
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, *objective_arguments), n_trials=optimization_steps)

            x_prime_hat = np.array(list(study.best_params.values()))
            prediction = model.predict_proba(x_prime_hat.reshape((1, -1)))[0][0]
            
            Y_primes.append(prediction)
            candidates.append(x_prime_hat)
    elif (optimization_method=="scipy"):
        boundaries = [boundaries[i][1:] for i in range(len(boundaries))]
        for lambda_k in lambdas:
            objective_arguments = x, y_prime_target, lambda_k, model, X
            # optimise the cost function -- assuming here it's smooth
            solution = scipy.optimize.minimize(cost_function, x, args=objective_arguments, 
                                               method='SLSQP', bounds=boundaries, options={'maxiter': optimization_steps})
            x_prime_hat = solution.x
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