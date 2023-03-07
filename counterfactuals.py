import scipy.stats
import numpy as np
import pandas as pd
from cost_functions import wachter2017_cost_function


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
        arguments = x, y_prime_target, lambda_k, model, X
        # optimise the cost function -- assuming here it's smooth
        solution = scipy.optimize.minimize(cost_function, 
                                           x, # start from our current observation
                                           args=arguments, method='SLSQP', bounds=bnds, constraints=cons)
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