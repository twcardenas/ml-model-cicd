import pandas as pd

training_data = pd.read_csv("train_split.csv", index_col=0)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=14)

predictors = ["venue_code", "opp_code", "hour", "day_code"]
rf.fit(training_data[predictors], training_data["target"])

import joblib
joblib.dump(rf, "./random_forest.joblib", compress=3)