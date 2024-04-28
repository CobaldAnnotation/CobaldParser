# CoBaLD Parser

CoBaLD parser is a neural network that allows one to annotate tokenized text (*.conllu file) in CoBaLD format.

## Usage

**Disclaimer**: the parser is implemented using AllenNLP framework, which is now in maintenance mode.
It causes certain compatibility problems, and one may need to put some efforts in order to make it work.
See [AllenNLP installation guide](https://docs.allennlp.org/v2.10.1/#installation) for details.

### Inference

See https://huggingface.co/CoBaLD for pretrained models and how to use them. 

### Train
```
./train_multistage.sh configs/<SELECT_JSONNET_CONFIG> <SERIALIZATION_DIR>
```
