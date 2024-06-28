# CoBaLD Parser

CoBaLD parser is a neural network that allows one to annotate tokenized text (*.conllu file) in CoBaLD format.

## Setup

First, create conda environment with python3.8 and activate it.
```
conda create --name <ENV_NAME> python=3.8
conda activate <ENV_NAME>
```

Next install git & pip inside conda and install requirements using pip.
(yes, it is not recommended to mix pip and conda, but in this case there is no other way to make things work, as we are building allennlp from forked repository)
```
conda install git pip
pip install -r requirements.txt
```

## Usage

### Training

The training pipeline consists of two stages: pretraining on external conllu dataset with syntactic markup and finetuning on the target conllu file. This allows parser to learn more about syntactic structure of a sentence and significantly increases parser quality. The training pipeline is evoked with the following command:
```
./train_multistage.sh configs/<JSONNET_CONFIG> <SERIALIZATION_DIR>
```

The external and the target datasets can be downloaded using appropriate scripts from `data` directory.

### Inference

See https://huggingface.co/CoBaLD for pretrained models and how to use them.

Once output conllu file is produced, it may be processed with postprocessing.py (see data/ directory) script in order to restore range tokens (ones with x-y indices):
```
./postprocessing.py test.conllu model_output.conllu final_output.conllu
```
