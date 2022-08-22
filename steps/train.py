import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from steps.lib.variables import predictors

def run():
    training_data = pd.read_csv("./output/train_split.csv", index_col=0)

    mlflow.set_tracking_uri(uri="http://localhost:5000")
    mlflow.set_experiment("sports predictor")
    with mlflow.start_run(run_name="Sports predictor") as run:
        run_id = run.info.run_id

        rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=14)
        mlflow.log_param("n_estimators", rf.n_estimators)
        mlflow.log_param("min_samples_split", rf.min_samples_split)
        mlflow.log_param("random_state", rf.random_state)

        # log input features
        mlflow.set_tag("features", str(training_data[predictors].columns.values.tolist()))


        rf.fit(training_data[predictors], training_data["target"])

        # get model signature
        signature = infer_signature(
            model_input=training_data[predictors],
            model_output=rf.predict(training_data[predictors])
        )

        mlflow.sklearn.log_model(rf, "saved_models", signature=signature)
        print(f"Run Id: {run_id}")
        return run_id