import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# Cargar datos
matches = pd.read_csv('matchess.csv')
matches = matches.dropna(subset=['score'])

# Preprocesamiento de las puntuaciones
matches['score'] = matches['score'].str.extract(r'(\d+–\d+)', expand=False)
matches[['home_goals', 'away_goals']] = matches['score'].str.split('–', expand=True)
matches['home_goals'] = pd.to_numeric(matches['home_goals'])
matches['away_goals'] = pd.to_numeric(matches['away_goals'])

# Agregar estadísticas del equipo
local_summary = matches.groupby('home_team')['home_goals'].sum().reset_index()
local_summary.rename(columns={'home_team': 'team', 'home_goals': 'home_goals_total'}, inplace=True)

away_summary = matches.groupby('away_team')['away_goals'].sum().reset_index()
away_summary.rename(columns={'away_team': 'team', 'away_goals': 'away_goals_total'}, inplace=True)

local_conceded_summary = matches.groupby('home_team')['away_goals'].sum().reset_index()
local_conceded_summary.rename(columns={'home_team': 'team', 'away_goals': 'home_conceded'}, inplace=True)

away_conceded_summary = matches.groupby('away_team')['home_goals'].sum().reset_index()
away_conceded_summary.rename(columns={'away_team': 'team', 'home_goals': 'away_conceded'}, inplace=True)

# Merge de las estadísticas
team_summary = pd.merge(local_summary, away_summary, on='team', how='outer').fillna(0)
team_summary2 = pd.merge(local_conceded_summary, away_conceded_summary, on='team', how='outer').fillna(0)
team_summary3 = pd.merge(team_summary, team_summary2, on='team', how='outer').fillna(0)

team_summary3['goalsScored'] = team_summary3['home_goals_total'] + team_summary3['away_goals_total']
team_summary3['goalsConceded'] = team_summary3['home_conceded'] + team_summary3['away_conceded']
team_summary3 = team_summary3.set_index("team")

# Calcular el resultado de cada partido (Victoria local, Empate, Victoria visitante)
matches['result'] = matches.apply(
    lambda row: 1 if row['home_goals'] > row['away_goals'] else (0 if row['home_goals'] < row['away_goals'] else -1),
    axis=1
)

# Merge de las estadísticas históricas al DataFrame de partidos
matches = pd.merge(matches, team_summary3, left_on='home_team', right_index=True, how='left')
matches = pd.merge(matches, team_summary3, left_on='away_team', right_index=True, suffixes=('_home', '_away'), how='left')

# Características adicionales
matches['home_advantage'] = 1  # 1 para el equipo local, 0 para el visitante
matches['diff_goalsScored'] = matches['goalsScored_home'] - matches['goalsScored_away']
matches['diff_goalsConceded'] = matches['goalsConceded_home'] - matches['goalsConceded_away']

# Seleccionar características y objetivo
features = ['home_advantage', 'diff_goalsScored', 'diff_goalsConceded']
X = matches[features]
y = matches['result']  # Resultado del partido: 1, 0, -1

# Balancear clases usando SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Ajuste de hiperparámetros con GridSearchCV para la regresión logística
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularización inversa
    'solver': ['liblinear', 'saga'],  # Métodos de optimización
    'max_iter': [100, 200, 300]  # Número de iteraciones
}

# Inicializar y ajustar el modelo
logreg = LogisticRegression(random_state=42, multi_class='ovr')
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Mejor modelo
best_model = grid_search.best_estimator_

# Hacer predicciones
y_pred = best_model.predict(X_test)

# Evaluar el rendimiento
print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Exactitud: {accuracy_score(y_test, y_pred)}")
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))
