import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import numpy as np

data_2018 = pd.read_csv('HomeAwayResults_2018_2019.csv')
data_2019 = pd.read_csv('HomeAwayResults_2019_2020.csv')
data_2020 = pd.read_csv('HomeAwayResults_2020_2021.csv')
data_2021 = pd.read_csv('HomeAwayResults_2021_2022.csv')
data_2022 = pd.read_csv('HomeAwayResults_2022_2023.csv')
data_2023 = pd.read_csv('HomeAwayResults_2023_2024.csv')
data_2024 = pd.read_csv('HomeAwayResults_2024_2025.csv')

# data_2019['year'] = 2019
# data_2020['year'] = 2020
# data_2021['year'] = 2021
# data_2022['year'] = 2022
# data_2023['year'] = 2023
# data_2024['year'] = 2024


data = pd.concat([data_2018, data_2019, data_2020, data_2021, data_2022, data_2023, data_2024], ignore_index=True)

# print(data.head())
# print(data.info())

# features = [
#     'home_wins', 'home_ties', 'home_losses', 'home_goals_for', 'home_goals_against', 
#     'home_goal_diff', 'home_points', 'home_points_avg', 'home_xg_for', 'home_xg_against', 
#     'home_xg_diff', 'home_xg_diff_per90', 'away_wins', 'away_ties', 'away_losses', 
#     'away_goals_for', 'away_goals_against', 'away_goal_diff', 'away_points', 
#     'away_points_avg', 'away_xg_for', 'away_xg_against', 'away_xg_diff', 'away_xg_diff_per90'
# ]

features = [
    'home_points_avg', 'home_xg_for', 'home_xg_against', 'home_xg_diff', 'home_xg_diff_per90',
    'away_points_avg', 'away_xg_for', 'away_xg_against', 'away_xg_diff', 'away_xg_diff_per90'
]

target = 'rank'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=10, 
                min_samples_leaf=4, 
                random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

predicciones = pd.DataFrame({'Predicted Rank': np.round(y_pred, 0), 'Actual Rank': y_test})
print(predicciones.head(10))


