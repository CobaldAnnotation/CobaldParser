# CoBaLD Parser

A neural network parser that annotates tokenized text (*.conllu file) in CoBaLD format.

## Setup

Create virtual environment, install dependencies and the project itself:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install .
```

## Usage

### Train
```
python main.py train [--batch_size BATCH_SIZE] [--n_epochs N_EPOCHS] train_conllu_path val_conllu_path serialization_dir

positional arguments:
  train_conllu_path     Path to the training .conllu file.
  val_conllu_path       Path to the validation .conllu file.
  serialization_dir     Path to model serialization directory. Must be empty.

options:
  --batch_size BATCH_SIZE
                        Batch size for dataloaders.
  --n_epochs N_EPOCHS   Number of training epochs.
```
Example:
```
python main.py train data/train.conllu data/validation.conllu serialization/distilbert --batch_size=32 --n_epochs=10
```

### Predict
```
python main.py predict [--batch_size BATCH_SIZE] input_conllu_path output_conllu_path serialization_dir

positional arguments:
  input_conllu_path     Path to a conllu file to read unlabeled sentences from.
  output_conllu_path    Path to a conllu file to write predictions to.
  serialization_dir     Path to a serialization directory with saved model that will be used for inference.

options:
  --batch_size BATCH_SIZE
                        Batch size for test dataloader.
```
Example:
```
python main.py predict data/test_clean.conllu predictions/test.conllu serialization/distilbert --batch_size=64
```
