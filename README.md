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

The model can be inferenced using predict.sh script, which takes model.tag.gz, input conllu and output conllu as arguments.

Example:
```
./predict.sh serialization/distilbert-cobald-parser/model.tar.gz data/test_clean.conllu predictions/distilbert.conllu
```

For pretrained models see dedicated page: https://huggingface.co/CoBaLD.

