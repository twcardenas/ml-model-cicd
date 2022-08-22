from steps import train
from steps import evaluate

run_id = train.run()
evaluate.run(run_id=run_id)