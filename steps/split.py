import pandas as pd

matches = pd.read_csv("./output/processed_data.csv", index_col=0)

train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']

train.to_csv("./output/train_split.csv")
test.to_csv("./output/test_split.csv")