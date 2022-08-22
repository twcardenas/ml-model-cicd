mkdir -p ./output
python3 preprocess.py
python3 split.py
python3 pipeline.py