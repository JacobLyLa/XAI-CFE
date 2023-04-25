import math
import pickle
import warnings

import pandas as pd

'''
from cost_functions import (wachter2017_cost_function,
                            wachter2017_cost_function_ignore_categorical,
                            wachter2017_cost_function_with_categorical,
                            weighted_watcher_cost_function)
from counterfactuals import get_counterfactuals
from prefrences import (Feature, get_constant_weight_function,
                        get_pow_weight_function)

warnings.filterwarnings('ignore')
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
'''

pd.options.display.float_format = '{:.5f}'.format


def main():
    with open('pretrained_models/pipeline_adult.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv('datasets/adult.csv', na_values='?')
    df = df.drop(columns = ['fnlwgt'])
    df = df.dropna()

    #num_ix = df.select_dtypes(include=['int64', 'float64']).columns

    X = df.drop(columns=['income', 'education'])
    #x=X.loc[15].values
    #x = x.reshape((1, -1))
    sample_df = X.loc[12:12]
    print(sample_df)
    print(model.predict_proba(sample_df))

    ashley = {'age': 30,
          'workclass': 'Private',
          'educational-num': 10,
          'marital-status': 'Never-married',
          'occupation': 'Sales', # so she is a cashier, or independent consultant for Oriflame
          'relationship': 'Own-child',
          'race': 'White',
          'gender': 'Female',
          'capital-gain': 0,
          'capital-loss': 0,
          'hours-per-week': 25,
          'native-country': 'United-States'}
    ashley_df = pd.DataFrame(ashley, index=[0])
    print("Ashley df:")
    print(ashley_df)
    
    #cost = wachter2017_cost_function_ignore_categorical(x_prime=sample_df, x=sample_df, y_prime=0.9, lambda_value=1, model = model, X=X)
    #print(cost)
    print()
    features = []
    features.append(Feature('age', float, (30, 50), get_constant_weight_function(1)))
    features.append(Feature('workclass', object, X['workclass'].unique(), get_constant_weight_function(1)))
    features.append(Feature('educational-num', int, (10, 14), get_constant_weight_function(1)))
    features.append(Feature('marital-status', object, X['marital-status'].unique(), get_constant_weight_function(1)))
    features.append(Feature('occupation', object, X['occupation'].unique(), get_constant_weight_function(1)))
    features.append(Feature('relationship', object, X['relationship'].unique(), get_constant_weight_function(1)))
    features.append(Feature('race', object, X['race'].unique(), get_constant_weight_function(1.5)))
    features.append(Feature('gender', object, X['gender'].unique(), get_constant_weight_function(1)))
    features.append(Feature('capital-gain', int, (0, 10000), get_constant_weight_function(1)))
    features.append(Feature('capital-loss', int, (0, 1000), get_constant_weight_function(1)))
    features.append(Feature('hours-per-week', int, (0, 70), get_constant_weight_function(1)))
    features.append(Feature('native-country', object, X['native-country'].unique(), get_constant_weight_function(1)))


    counterfactuals = get_counterfactuals(x=ashley_df, y_prime_target=0.3, model=model, X=X, cost_function=wachter2017_cost_function_with_categorical, \
                                           features=features, tol=0.2, optimization_method="optuna", optimization_steps=100, framed=True)
    print(model.predict_proba(ashley_df))
    print("Ashley Counterfactuals:")
    print(counterfactuals)
    print(model.predict_proba(counterfactuals))
    

if __name__ == "__main__":
    main()
    # categories