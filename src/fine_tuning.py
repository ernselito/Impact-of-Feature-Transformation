from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def hyperparameter_tuning(X, y):
    param_grid = [
        {'n_estimators': [3, 10, 30], 'bootstrap': [False], 'max_features': [2, 4, 6, 8]},
        {'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X, y)
    return grid_search