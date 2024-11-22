import pandas as pd

matches = pd.read_csv('matches.csv')

matches = matches.dropna(subset=['score'])
matches[['home_goals']] = matches['score'].str.split('–', expand=True)

print(matches)