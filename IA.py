import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split

matches = pd.read_csv('matchess.csv')

matches = matches.dropna(subset=['score'])

matches['score'] = matches['score'].str.extract(r'(\d+–\d+)', expand=False)

matches[['home_goals', 'away_goals']] = matches['score'].str.split('–', expand=True)

matches['home_goals'] = pd.to_numeric(matches['home_goals'])
matches['away_goals'] = pd.to_numeric(matches['away_goals'])

local_summary = matches.groupby('home_team')['home_goals'].sum().reset_index()
local_summary.rename(columns={'home_team': 'team', 'home_goals': 'home_goals_total'}, inplace=True)

away_summary = matches.groupby('away_team')['away_goals'].sum().reset_index()
away_summary.rename(columns={'away_team': 'team', 'away_goals': 'away_goals_total'}, inplace=True)

local_conceded_summary = matches.groupby('home_team')['away_goals'].sum().reset_index()
local_conceded_summary.rename(columns={'home_team': 'team', 'away_goals': 'home_conceded'}, inplace=True)

away_conceded_summary = matches.groupby('away_team')['home_goals'].sum().reset_index()
away_conceded_summary.rename(columns={'away_team': 'team', 'home_goals': 'away_conceded'}, inplace=True)

team_summary = pd.merge(local_summary, away_summary, on='team', how='outer').fillna(0)
team_summary2 = pd.merge(local_conceded_summary, away_conceded_summary, on='team', how='outer').fillna(0)
team_summary3 = pd.merge(team_summary, team_summary2, on='team', how='outer').fillna(0)

team_summary3['home_goals_total'] = team_summary3['home_goals_total'].astype(int)
team_summary3['away_goals_total'] = team_summary3['away_goals_total'].astype(int)
team_summary3['home_conceded'] = team_summary3['home_conceded'].astype(int)
team_summary3['away_conceded'] = team_summary3['away_conceded'].astype(int)

team_summary3['goalsScored'] = team_summary3['home_goals_total'] + team_summary3['away_goals_total']
team_summary3['goalsConceded'] = team_summary3['home_conceded'] + team_summary3['away_conceded']

team_summary3 = team_summary3.set_index("team")
print(team_summary3)
print(matches)

matches['result'] = matches.apply(
    lambda row: 1 if row['home_goals'] > row['away_goals'] else (0 if row['home_goals'] < row['away_goals'] else -1),
    axis=1
)

matches.to_csv('matches2.csv', index=False, encoding="UTF-8")


def calculate_form(df, team, n=5):
    recent_matches = df[(df['home_team'] == team) | (df['away_team'] == team)]
    points = 0
    goals_scored = 0
    goals_conceded = 0
    for _, match in recent_matches.iterrows():
        if match['home_team'] == team:
            goals_scored += match['home_goals']
            goals_conceded += match['away_goals']
            points += 3 if match['home_goals'] > match['away_goals'] else 1 if match['home_goals'] == match['away_goals'] else 0
        else:
            goals_scored += match['away_goals']
            goals_conceded += match['home_goals']
            points += 3 if match['away_goals'] > match['home_goals'] else 1 if match['away_goals'] == match['home_goals'] else 0
    return points, goals_scored / len(recent_matches), goals_conceded / len(recent_matches)

# Ejemplo para el equipo América
team = "América"
points, avg_goals_scored, avg_goals_conceded = calculate_form(matches, team, n=5)
print(f"{team} - Puntos: {points}, Promedio goles anotados: {avg_goals_scored}, Promedio goles recibidos: {avg_goals_conceded}")

def head_to_head(df, home_team, away_team):
    matches = df[((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
                 ((df['home_team'] == away_team) & (df['away_team'] == home_team))]
    total_matches = matches.shape[0]
    home_wins = matches[(matches['home_team'] == home_team) & (matches['home_goals'] > matches['away_goals'])].shape[0]
    away_wins = matches[(matches['away_team'] == away_team) & (matches['away_goals'] > matches['home_goals'])].shape[0]
    draws = matches[(matches['home_goals'] == matches['away_goals'])].shape[0]
    return total_matches, home_wins, away_wins, draws

# Ejemplo para América vs Chivas
home_team = "América"
away_team = "Guadalajara"
h2h_stats = head_to_head(matches, home_team, away_team)
print(f"Partidos totales: {h2h_stats[0]}, Victorias {home_team}: {h2h_stats[1]}, Victorias {away_team}: {h2h_stats[2]}, Empates: {h2h_stats[3]}")

# Merge de características históricas al DataFrame principal
matches = pd.merge(matches, team_summary3, left_on='home_team', right_index=True, how='left')
matches = pd.merge(matches, team_summary3, left_on='away_team', right_index=True, suffixes=('_home', '_away'), how='left')
matches.to_csv('matches4.csv', index=False, encoding="UTF-8")
print(matches)

matches['home_advantage'] = 1  # 1 para el equipo local, 0 para el visitante
matches['diff_goalsScored'] = matches['goalsScored_home'] - matches['goalsScored_away']
matches['diff_goalsConceded'] = matches['goalsConceded_home'] - matches['goalsConceded_away']

# Seleccionar características y objetivo
features = ['home_advantage', 'diff_goalsScored', 'diff_goalsConceded']
X = matches[features]
y = matches['result']  # Resultado del partido: 1, 0, -1

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")



home_team = "Guadalajara"
away_team = "Atlas"

# Obtener estadísticas históricas de los equipos
home_stats = team_summary3.loc[home_team]
away_stats = team_summary3.loc[away_team]

# Calcular características
home_advantage = 1
diff_goalsScored = home_stats['goalsScored'] - away_stats['goalsScored']
diff_goalsConceded = home_stats['goalsConceded'] - away_stats['goalsConceded']

# Crear DataFrame para el modelo
match_features = pd.DataFrame([{
    'home_advantage': home_advantage,
    'diff_goalsScored': diff_goalsScored,
    'diff_goalsConceded': diff_goalsConceded
}])

# Hacer predicción
prediction = model.predict(match_features)[0]

# Interpretar el resultado
result_map = {1: 'Victoria local', 0: 'Empate', -1: 'Victoria visitante'}
print(f"Resultado predicho: {result_map[prediction]}")




# print(matches)
# print(team_summary3)
# print(team_summary3.at["América", 'goalsScored'])