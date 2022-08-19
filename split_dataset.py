import pandas as pd
matches = pd.read_csv("processed_data.csv", index_col=0)

train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']

train.to_csv("./train_split.csv")
test.to_csv("./test_split.csv")