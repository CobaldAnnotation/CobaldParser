# CoBaLD Parser

A neural network parser that annotates tokenized text (*.conllu file) in CoBaLD format.

## Setup

Using python virtual enviroment:
```
python -m venv .venv
source .venv/bin/activate
pip install .
```

## Usage

### Train

Example:
```
python train.py \
    --model_config model_config.json \
    --dataset_path CoBaLD/enhanced-cobald-dataset \
    --dataset_name en \
    --output_dir serialization/distilbert-en \
    --push_to_hub \
    --hub_model_id CoBaLD/cobald-parser-distilbert-en
```

For a full list of options, run `python train.py -h`.

### Predict

**TODO**