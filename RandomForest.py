import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data_folder = os.path.join("Data", "basketball")
data_filename = os.path.join(data_folder, "leagues_NBA_2017_games_games.csv")
results1 = pd.read_csv(data_filename)
results1.ix[:5]
results = pd.read_csv(data_filename, parse_dates=["Date"], skiprows=[0])
results.columns = ["Date","Time", "Score Type", "Visitor Team", "VisitorPts", "Home Team", "HomePts", "OT?", "Notes"]
results.ix[:5]
results["HomeWin"] = results["VisitorPts"] < results["HomePts"]
y_true = results["HomeWin"].values
results["HomeLastWin"] = False
results["VisitorLastWin"] = False


won_last = defaultdict(int)

for index, row in results.iterrows():  
    home_team = row["Home Team"]  
    visitor_team = row["Visitor Team"]  
    row["HomeLastWin"] = won_last[home_team] 
    row["VisitorLastWin"] = won_last[visitor_team] 
    results.ix[index] = row  
    
    won_last[home_team] = row["HomeWin"] 
    won_last[visitor_team] = not row["HomeWin"]
results.ix[20:25]

clf = DecisionTreeClassifier(random_state=14)

X_previouswins = results[["HomeLastWin", "VisitorLastWin"]].values
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_previouswins, y_true, scoring='accuracy')

ladder_filename = os.path.join(data_folder, "leagues_NBA_2017_standings_expanded-standings.csv")
ladder = pd.read_csv(ladder_filename, skiprows=[0,1])
ladder
results["HomeTeamRanksHigher"] = 0

for index, row in results.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    if home_team == "New Orleans Pelicans":
        home_team = "New Orleans Hornets"
    elif visitor_team == "New Orleans Pelicans":
        visitor_team = "New Orleans Hornets"
    home_rank = ladder[ladder["Team"] == home_team]["Rk"].values[0]
    visitor_rank = ladder[ladder["Team"] == visitor_team]["Rk"].values[0]
    row["HomeTeamRanksHigher"] = int(home_rank > visitor_rank)
    results.ix[index] = row
results[:5]
X_homehigher =  results[["HomeLastWin", "VisitorLastWin", "HomeTeamRanksHigher"]].values
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_homehigher, y_true, scoring='accuracy')
print("Using whether the home team is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

from sklearn.grid_search import GridSearchCV
parameter_space = {
                   "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                   }
clf = DecisionTreeClassifier(random_state=14)
grid = GridSearchCV(clf, parameter_space)
grid.fit(X_homehigher, y_true)
print("Accuracy: {0:.1f}%".format(grid.best_score_ * 100))
last_match_winner = defaultdict(int)
results["HomeTeamWonLast"] = 0

for index, row in results.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    teams = tuple(sorted([home_team, visitor_team]))      
    row["HomeTeamWonLast"] = 1 if last_match_winner[teams] == row["Home Team"] else 0
    results.ix[index] = row
    winner = row["Home Team"] if row["HomeWin"] else row["Visitor Team"]
    last_match_winner[teams] = winner
results.ix[:5]
X_home_higher =  results[["HomeTeamRanksHigher", "HomeTeamWonLast"]].values
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_home_higher, y_true, scoring='accuracy')

encoding = LabelEncoder()
encoding.fit(results["Home Team"].values)
home_teams = encoding.transform(results["Home Team"].values)
visitor_teams = encoding.transform(results["Visitor Team"].values)

X_teams = np.vstack([home_teams, visitor_teams]).T
onehot = OneHotEncoder()

X_teams = onehot.fit_transform(X_teams).todense()
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_teams, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_teams, y_true, scoring='accuracy')
print("Using full team labels is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))
X_all = np.hstack([X_home_higher, X_teams])

clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_all, y_true, scoring='accuracy')

print("Using whether the home team is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

parameter_space = {
                   "max_features": [2, 10, 'auto'],
                   "n_estimators": [100,],
                   "criterion": ["gini", "entropy"],
                   "min_samples_leaf": [2, 4, 6],
                   }

clf = RandomForestClassifier(random_state=14, n_jobs=-1)
grid = GridSearchCV(clf, parameter_space)
grid.fit(X_all, y_true)

print("Accuracy: {0:.1f}%".format(grid.best_score_ * 100))
print(grid.best_estimator_)
