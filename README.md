# ml-model-cicd
Test splitting a Random Forrest Classifier into a automated workflow

# Objective
Take this notebook (https://github.com/dataquestio/project-walkthroughs/blob/master/football_matches/prediction.ipynb) and convert the model to something that can be deployed and iterated on.

Todo this I will separate the notebook into automated workflows:
* Data Pre-Processing
* Training
* Model Evaluation

Additional components will be a model registry. That way we can keep track of our models and dockerize models that exist in the registry.

# Todo

* Make diagram of flow
* Set Up MLFlow
* Unit tests

# Model Inputs & Outputs
Input:
```
{
    venue_code: Integer,
    opp_code: Integer,
    hour: Integer,
    day_code: Integer
}
```

Output: 0 or 1

# Workflow

1. preprocess.py
Load data sources, format, transform. Then save new dataset

2. split_dataset.py
Split the processed dataset into two, one to fit the model and one to test the model.
We will train on previous seasons and predict on the current / latest season

3. train.py
Load training data set and fit the model. save the model output

4. evaluate.py

Load a saved Model and test dataset for accuracy
