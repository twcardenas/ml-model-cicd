import pandas as pd
import joblib

predictors = ["venue_code", "opp_code", "hour", "day_code"]

input = pd.read_csv('./test_split.csv', index_col=0)
# load, no need to initialize the loaded_rf
rf = joblib.load("./random_forest.joblib")

preds = rf.predict(input[predictors])

from sklearn.metrics import accuracy_score
error = accuracy_score(input["target"], preds)
print(f"Error: {error}")

combined = pd.DataFrame(dict(actual=input["target"], predicted=preds))
pd.crosstab(index=combined["actual"], columns=combined["predicted"])

from sklearn.metrics import precision_score

print(precision_score(input["target"], preds))