import pandas as pd
from steps.lib.variables import predictors

def run(run_id:str):
    input = pd.read_csv('./output/test_split.csv', index_col=0)
    import mlflow
    mlflow.set_tracking_uri(uri="http://localhost:5000")
    import mlflow.sklearn
    rf = mlflow.sklearn.load_model(f"runs:/{run_id}/saved_models")

    with mlflow.start_run(run_id=run_id) as run:
        preds = rf.predict(input[predictors])

        from sklearn.metrics import accuracy_score
        error = accuracy_score(input["target"], preds)
        print(f"Error: {error}")
        mlflow.log_metric("accuracy_score", error)

        combined = pd.DataFrame(dict(actual=input["target"], predicted=preds))
        pd.crosstab(index=combined["actual"], columns=combined["predicted"])

        from sklearn.metrics import precision_score

        precision = precision_score(input["target"], preds)
        mlflow.log_metric("precision_score", precision)