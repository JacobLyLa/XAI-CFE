import scipy.stats
import numpy as np
import pandas as pd
from cost_functions import wachter2017_cost_function
import optuna
# turn off optuna warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, x, y_prime_target, lambda_k, model, X, cost_function, features):
    for feature in features:
        feature.sample(trial)

    x_prime = np.array([feature.value for feature in features])
    weight_functions = [feature.weight_function for feature in features]
    return cost_function(x_prime, x, y_prime_target, lambda_k, model, X, weight_functions)

def get_counterfactuals(x, y_prime_target, model, X, cost_function, features, tol=0.05, optimization_method="optuna", optimization_steps=10, framed=True):    
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
            objective_arguments = x, y_prime_target, lambda_k, model, X, cost_function, features
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, *objective_arguments), n_trials=optimization_steps)

            x_prime_hat = np.array(list(study.best_params.values()))
            prediction = model.predict_proba(x_prime_hat.reshape((1, -1)))[0][0]
            
            Y_primes.append(prediction)
            candidates.append(x_prime_hat)
    elif (optimization_method=="scipy"):
        bounds = [features[i].boundaries for i in range(len(features))]
        for lambda_k in lambdas:
            weight_functions = [feature.weight_function for feature in features]
            objective_arguments = x, y_prime_target, lambda_k, model, X, weight_functions
            # optimise the cost function -- assuming here it's smooth
            solution = scipy.optimize.minimize(cost_function, x, args=objective_arguments, 
                                               method='SLSQP', bounds=bounds, options={'maxiter': optimization_steps})
            x_prime_hat = solution.x
            prediction = model.predict_proba(x_prime_hat.reshape((1, -1)))[0][0]
            Y_primes.append(prediction)
            candidates.append(x_prime_hat)
    Y_primes = np.array(Y_primes)
    candidates = np.array(candidates)
    # check if any counterfactual candidates meet the tolerance condition
    eps_condition = np.abs(Y_primes - y_prime_target) <= tol
    relevant_candidates = candidates[eps_condition]
    # print y_primes of relevant candidates
    print(Y_primes[eps_condition])
    if framed:
        return pd.DataFrame(data=relevant_candidates, columns=X.columns)
    return relevant_candidates