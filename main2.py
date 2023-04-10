import pickle
import pandas as pd
from counterfactuals import get_counterfactuals
from cost_functions import wachter2017_cost_function, weighted_watcher_cost_function, wachter2017_cost_function_ignore_categorical, \
            wachter2017_cost_function_with_categorical
import warnings
import math
from prefrences import Feature, get_constant_weight_function, get_pow_weight_function
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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
    #sample_df = X.loc[12:12]
    #print(sample_df)
    #print(model.predict_proba(sample_df))
    #print(model.predict_proba(sample_df))

    ashley = {'age': 30,
          'workclass': 'Private',
          'educational-num': 10,
          'marital-status': 'Never-married',
          'occupation': 'Sales', # so she is a cashier, or ndependent consultant for Oriflame
          'relationship': 'Own-child',
          'race': 'White',
          'gender': 'Female',
          'capital-gain': 0,
          'capital-loss': 0,
          'hours-per-week': 25,
          'native-country': 'United-States'}
    ashley_df = pd.DataFrame(ashley, index=[0])
  #  cost = wachter2017_cost_function_ignore_categorical(x_prime=sample_df, x=sample_df, y_prime=0.9, lambda_value=1, model = model, X=X)
   # print(cost)
    features = []
    features.append(Feature('age', float, (0, 100), get_constant_weight_function(1)))
    features.append(Feature('workclass', object, ['Private', 'Local-gov', 'Self-emp-not-inc', 'Federal-gov',
                                                  'State-gov', 'Self-emp-inc', 'Without-pay'], get_constant_weight_function(1)))
    features.append(Feature('educational-num', int, (1, 14), get_constant_weight_function(1)))
    features.append(Feature('capital-gain', int, (ashley_df.values[0][8]-20, ashley_df.values[0][8]+20), get_constant_weight_function(1)))
    features.append(Feature('capital-loss', int, (ashley_df.values[0][9]-20, ashley_df.values[0][9]+20), get_constant_weight_function(1)))
    features.append(Feature('hours-per-week', int, (0, 100), get_constant_weight_function(1)))
    counterfactuals = get_counterfactuals(x=ashley_df, y_prime_target=0.3, model=model, X=X, cost_function=wachter2017_cost_function_with_categorical, \
                                           features=features, tol=0.5, optimization_method="optuna", optimization_steps=100, framed=True)

    print(counterfactuals)
    #print(model.predict_proba(counterfactuals))
    

if __name__ == "__main__":
    main()
    # categories