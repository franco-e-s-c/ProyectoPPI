import pandas as pd

matches = pd.read_csv('matches.csv')

matches = matches.dropna(subset=['score'])
<<<<<<< Updated upstream
matches[['home_goals']] = matches['score'].str.split('–', expand=True)
=======

matches['score'] = matches['score'].str.extract(r'(\d+–\d+)', expand=False)
>>>>>>> Stashed changes

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

matches.to_csv('matches2.csv', index=False, encoding="UTF-8")

print(matches)
print(team_summary3)