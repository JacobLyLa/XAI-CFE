import pickle
import pandas as pd
from counterfactuals import get_counterfactuals
from cost_functions import wachter2017_cost_function
import warnings
warnings.filterwarnings('ignore')

def main():
    with open('pretrained_models/model_svc_diabetes.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv('datasets/diabetes.csv')

    # only first 2 features
    X = df.drop(columns = 'Outcome')
    #y = df['Outcome']

    P_prime = 0.2
    bnds = ((0,None), (100,200), (50, 200), (20, 50), (0, None), (10, 100), (0, None), (0, 100))
    cons = ({'type':'eq','fun': lambda x :  x[7]-int(x[7])})
    counterfactuals = get_counterfactuals(X.loc[1].values, P_prime, model, X, 
                                          cost_function=wachter2017_cost_function, tol=0.05, bnds=bnds)
    print("original example:")
    print(X.loc[1])
    print(model.predict_proba(X.loc[1].values.reshape((1, -1))))
    print(counterfactuals)
    differences = counterfactuals.values - X.loc[1].values
    differences_df = pd.DataFrame(data=differences, columns=X.columns)
    print("Differences:")
    print(differences_df)

if __name__ == "__main__":
    main()