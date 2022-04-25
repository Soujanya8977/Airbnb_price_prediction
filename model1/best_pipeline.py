import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('Airbnb_price_prediction/model1/prepared_data.csv')
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -48683.42564645546
exported_pipeline = DecisionTreeRegressor(max_depth=9, min_samples_leaf=14, min_samples_split=10)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
