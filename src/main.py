import pandas as pd
import numpy as np
from src.data_loader_and_preprocessing import load_data, preprocess_housing_data, stratified_split
from src.feature_engineering import create_new_features, AdvancedFeatureEngineer
from src.modeling import train_and_compare_models, models
from src.hyperparameter_tuning import hyperparameter_tuning

# Loading data
housing = load_data()

# Preprocessing data
housing_preprocessed = preprocess_housing_data(housing)

# Spliting data
strat_splits = stratified_split(housing)

# Creating new features
housing_basic = create_new_features(housing_preprocessed)

# Advance feature engineering
feature_engineer = AdvancedFeatureEngineer()
housing_advanced = feature_engineer.fit_transform(housing_preprocessed)

# Training and comparing models
train_and_compare_models(models, housing_preprocessed, housing_labels, housing_test_baseline, housing_labels_test)

# Hyperparameter tuning
grid_search = hyperparameter_tuning(housing_advanced, housing_labels)