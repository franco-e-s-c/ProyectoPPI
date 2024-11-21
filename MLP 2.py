import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import GridSearchCV

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

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 128, 64, 32),
                   max_iter=10000,
                   random_state=42,
                   learning_rate_init=0.001,
                   alpha=0.01,
                   momentum=0.8,
                   activation='relu',
                   solver='sgd')

mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
predicciones = pd.DataFrame({'Predicted Rank': np.round(y_pred, 0), 'Actual Rank': y_test})
print(predicciones.head(20))
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

tolerance = 1
correct_predictions = np.abs(np.round(y_pred) - y_test) <= tolerance
accuracy = np.mean(correct_predictions) * 100

print(f"Overall Accuracy (tolerance <= {tolerance}): {accuracy:.2f}%")

