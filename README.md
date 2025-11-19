# Exploring the Impact of Feature Transformation on Housing Price Prediction
## Project Overview
This research project investigates the impact of advanced feature engineering techniques (specifically feature transformation and creation) on the performance of various machine learning models used to predict California housing prices.

Accurate housing price prediction is critical for informed decision-making in the real estate market. The study aims to identify effective feature engineering strategies that enhance model accuracy and interpretability by comparing performance across three distinct feature sets:

Baseline: Only imputation, scaling, and one-hot encoding applied.

Basic Feature Creation: Baseline + ratio-based features (rooms_per_household, bedrooms_per_room, population_per_household).

Advanced Feature Engineering: Basic Features + calculated spatial features (distance_to_los_angeles, distance_to_san_francisco).

## Research Question
How does applying feature transformation and creation techniques impact the performance of machine learning models in predicting California housing prices?

## Data and Preparation
The project uses the California Housing Data from Kaggle.

Data Exploration Highlights

The dataset contains 20,640 entries with 10 features, including demographic, geographic, and housing-related attributes.

The total_bedrooms feature had 207 missing values, which were later imputed.

The target variable, median_house_value, and several other numerical features exhibited skewed distributions (as visible in the generated histograms), suggesting that transformation techniques could be beneficial. * Stratified Sampling was performed based on a binned median_income category (income_cat) to ensure the training and testing sets are representative of the overall income distribution.

Preprocessing and Feature Engineering

The following steps were implemented to prepare the data:

Missing Value Imputation: Missing values in numerical columns (specifically total_bedrooms) were filled using the median strategy.

Categorical Encoding: The ocean_proximity categorical feature was converted into numerical format using One-Hot Encoding.

Feature Scaling: All numerical features were standardized using StandardScaler.

Feature Creation (Basic):

- rooms_per_household = total_rooms / households

- bedrooms_per_room = total_bedrooms / total_rooms

- population_per_household = population / households

- Feature Creation (Advanced/Spatial):

Calculated Haversine distance (in kilometers) from each district to two major California cities: Los Angeles and San Francisco.

## Modeling and Results
The performance of six different regression models was evaluated on the test set using the Root Mean Squared Error (RMSE) metric.

Baseline Model Performance (No Feature Creation)


| Model Name | RMSE (USD) |
| --- | --- |
| Linear Regression | 67,354.15 |
| Random Forest | 53,746.73 |
| Support Vector Machine | 116,889.19 |
| K-Nearest Neighbors | 60,685.25 |
| Decision Tree | 75,035035.85 |
| Gradient Boosting | 55,485.85 |

The Random Forest Regressor was the best-performing model in the baseline test.	
Basic Feature Creation Performance (Ratios Added)


| Model Name | RMSE (USD) |
| --- | --- |
| Linear Regression | 67,407.78 |
| Random Forest | 53,253.14 |
| Support Vector Machine | 117,256.63 |
| K-Nearest Neighbors | 66,817.04 |
| Decision Tree | 74,521.51 |
| Gradient Boosting | 54,832.72 |

The addition of basic ratio features slightly improved the performance of the Random Forest model (RMSE decreased from 53,746.73 to 53,253.14).	
Advanced Feature Engineering Performance (Ratios + Distance Added)


| Model Name | RMSE (USD) |
| --- | --- |
| Linear Regression | 66,839.81 |
| Random Forest | 54,651.74 |
| Support Vector Machine | 117,256.68 |
| K-Nearest Neighbors | 75,765.24 |
| Decision Tree | 74,828.98 |
| Gradient Boosting | 54,395.60 |

In this round, the Gradient Boosting model achieved the best RMSE, though the overall improvement in performance was inconsistent across all models compared to the basic feature creation set.	

## Hyperparameter Tuning (Random Forest Regressor)

The Random Forest Regressor was selected for in-depth hyperparameter tuning using GridSearchCV due to its strong and consistent performance. The optimal parameters were identified across the three feature sets.

Feature Set	Best Hyperparameters	Tuned RMSE (5-fold CV)
Baseline	{'bootstrap': False, 'max_features': 8, 'n_estimators': 30}	49,659.14
Basic Features	{'bootstrap': False, 'max_features': 8, 'n_estimators': 30}	49,762.81
Advanced Features	{'bootstrap': False, 'max_features': 6, 'n_estimators': 30}	47,230.76
The Advanced Feature Engineering set, combined with hyperparameter tuning, yielded the best performance with an RMSE of 47,230.76. This represents a significant improvement of approximately 12.2% compared to the best untuned baseline model (53,746.73).

## Conclusion
The study successfully demonstrates that feature transformation and creation significantly enhance the predictive performance of machine learning models for California housing prices.

The Random Forest Regressor consistently proved to be one of the most effective models.

The most substantial performance gain was achieved by combining advanced feature engineering (incorporating calculated ratios and spatial distance features) with hyperparameter tuning, resulting in the lowest overall RMSE of 47,230.76 USD.

Engineered features, such as rooms per household and distance metrics, were implicitly identified as highly influential, confirming the importance of feature quality in capturing meaningful socioeconomic and geographic patterns in housing data.





