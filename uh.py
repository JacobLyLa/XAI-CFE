import tensorflow as tf
import numpy as np
from sklearn.datasets import load_diabetes
from witwidget.notebook.visualization import WitWidget
from witwidget.notebook.visualization import WitConfigBuilder

# load diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# define the XGBoost model
import xgboost as xgb
params = {
    "n_estimators": 100,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}
model = xgb.XGBRegressor(**params)
model.fit(X, y)

# define the feature names and categories
feature_names = diabetes.feature_names
categories = ['quantitative'] * X.shape[1]

# define the instance to be explained
instance = X[0]

# define the target class
target_class = 1

# define the counterfactual explainer using WIT
config_builder = WitConfigBuilder(instance.tolist(), feature_names=feature_names)
config_builder.set_model_type(model)
config_builder.set_target_feature(target_class)
config_builder.set_custom_predict_fn(lambda instances: model.predict(np.array(instances)))
config_builder.set_counterfactual_enabled(True)
wit_config = config_builder.build()
WitWidget(wit_config)
