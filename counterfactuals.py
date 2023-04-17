import scipy.stats
import numpy as np
import pandas as pd
from cost_functions import wachter2017_cost_function
import optuna
# turn off optuna warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, x, y_prime_target, lambda_k, model, X, cost_function, features):
    x_prime = x.copy()
    for feature in features:
        feature.sample(trial)
        # I tilfelle vi ikke sender alle features (fordi vi sender bare de som vi vil endre, derfor først koperer x, så endrer bare på de aktuelle)
        x_prime.at[x_prime.index[0], feature.name] = feature.value
    #x_prime = np.array([feature.value for feature in features])
    
    
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
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler( seed=42))
            study.optimize(lambda trial: objective(trial, *objective_arguments), n_trials=optimization_steps)

            #x_prime_hat = np.array(list(study.best_params.values()))
            # i tilfelle ikke endrer alle features, sender bare de som skal endres hit, så endrer på de aktuelle
            x_prime_hat = x.copy() #

            for param in x.columns.values:
                if param in study.best_params:
                    #x_prime_hat[param] = study.best_params[param]
                    x_prime_hat.at[x_prime_hat.index[0], param] = study.best_params[param]
            try:        
                prediction = model.predict_proba(x_prime_hat.reshape((1, -1)))[0][0] #when the model is train to predict from np.array
            except:
                prediction = model.predict_proba(x_prime_hat)[0][0] # when the model is train to predict from DataFrane
            Y_primes.append(prediction)
            candidates.append(x_prime_hat.values[0]) #siden nå sender som dataframe
    elif (optimization_method=="scipy"):
        bounds = [features[i].boundaries for i in range(len(features))]
        for lambda_k in lambdas:
            weight_functions = [feature.weight_function for feature in features]
            objective_arguments = x, y_prime_target, lambda_k, model, X, weight_functions
            # optimise the cost function -- assuming here it's smooth
            opt_features = [features[i].name for i in range(len(features))] # features that will be used in optimization
            solution = scipy.optimize.minimize(cost_function, x[opt_features], args=objective_arguments,
                                            method='SLSQP', bounds=bounds, options={'maxiter': optimization_steps})
                              
            x_prime_hat = solution.x
            x_prime_hat_to_df = x.copy()
            for i in range(len(x_prime_hat)):
                x_prime_hat_to_df.at[x.index[0],opt_features[i]] = x_prime_hat[i] #create dataframe where the values from x_prime is used 
            
            try:
                prediction = model.predict_proba(x_prime_hat.reshape((1, -1)))[0][0] #for the model which is trained on dataframe (adult income model)
            except:
                prediction = model.predict_proba( x_prime_hat_to_df)[0][0]
            Y_primes.append(prediction)
            #candidates.append(x_prime_hat)
            candidates.append(x_prime_hat_to_df.values[0])
    Y_primes = np.array(Y_primes)
    candidates = np.array(candidates)
    # check if any counterfactual candidates meet the tolerance condition
    eps_condition = np.abs(Y_primes - y_prime_target) <= tol
    relevant_candidates = candidates[eps_condition]
    if framed and len(relevant_candidates)>0:
        return pd.DataFrame(data=relevant_candidates, columns=X.columns)
    return relevant_candidates