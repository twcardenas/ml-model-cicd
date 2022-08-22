import pandas as pd
matches = pd.read_csv("./datasourcs/matches.csv", index_col=0)

# Data Pre-Processing
matches[matches["team"] == "Liverpool"].sort_values("date")
del matches["comp"]
del matches["notes"]

matches["date"] = pd.to_datetime(matches["date"])
matches["target"] = (matches["result"] == "W").astype("int")
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")

matches["day_code"] = matches["date"].dt.dayofweek

matches.to_csv("./output/processed_data.csv")