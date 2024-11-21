import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import numpy as np



# data_2019['year'] = 2019
# data_2020['year'] = 2020
# data_2021['year'] = 2021
# data_2022['year'] = 2022
# data_2023['year'] = 2023
# data_2024['year'] = 2024

# print(data.head())
# print(data.info())

# features = [
#     'home_wins', 'home_ties', 'home_losses', 'home_goals_for', 'home_goals_against', 
#     'home_goal_diff', 'home_points', 'home_points_avg', 'home_xg_for', 'home_xg_against', 
#     'home_xg_diff', 'home_xg_diff_per90', 'away_wins', 'away_ties', 'away_losses', 
#     'away_goals_for', 'away_goals_against', 'away_goal_diff', 'away_points', 
#     'away_points_avg', 'away_xg_for', 'away_xg_against', 'away_xg_diff', 'away_xg_diff_per90'
# ]

ap2019 = pd.read_csv('apertura_2019.csv')
clau2020 = pd.read_csv('clausura_2020.csv')
ap2020 = pd.read_csv('apertura_2020.csv')
clau2021 = pd.read_csv('clausura_2021.csv')
ap2021 = pd.read_csv('apertura_2021.csv')
clau2022 = pd.read_csv('clausura_2022.csv')
ap2022 = pd.read_csv('apertura_2022.csv')
clau2023 = pd.read_csv('clausura_2023.csv')
ap2023 = pd.read_csv('apertura_2023.csv')
clau2024 = pd.read_csv('clausura_2024.csv')
ap2024 = pd.read_csv('apertura_2024.csv')

data = pd.concat([ap2019, clau2020, ap2020, clau2021, ap2021, clau2022, ap2022, clau2023, ap2023, clau2024, ap2024], ignore_index=True)

features = [
    'xg_for', 'xg_against', 'xg_diff', 'xg_diff_per90'
]

target = 'rank'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(
                n_estimators=400, 
                max_depth=20, 
                min_samples_split=10, 
                min_samples_leaf=4, 
                random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

predicciones = pd.DataFrame({'Predicted Rank': np.round(y_pred, 0), 'Actual Rank': y_test})
print(predicciones.head(10))
print(f"Mean Absolute Error (MAE): {mae}")


