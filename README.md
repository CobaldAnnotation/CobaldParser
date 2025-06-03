# CoBaLD Parser
[Paper](https://dialogue-conf.org/wp-content/uploads/2025/04/BaiukIBaiukAPetrovaM.009.pdf)

A neural-based joint morphosyntactic and semantic parser capable of automatic annotation both in E-UD and in CoBaLD, including ellipsis restoration.

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
    --dataset_config_name en \
    --output_dir serialization/distilbert-en \
    --push_to_hub \
    --hub_model_id CoBaLD/cobald-parser-distilbert-en
```

For a full list of options, run `python train.py -h`.

### Predict

The model is integrated into huggingface pipeline ecosystem, which means no `git clone` required for model inference, just an installed `transformers` library. 

Example with NLTK tokenizer/sentenizer:
```py
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import pipeline

nltk.download('punkt_tab')

sentenizer = lambda text: sent_tokenize(text, "english")
tokenizer = lambda sentence: word_tokenize(sentence, preserve_line=True)
pipe = pipeline(
    "token-classification",
    model="CoBaLD/xlm-roberta-base-cobald-parser",
    trust_remote_code=True,
    sentenizer=sentenizer,
    tokenizer=tokenizer
)

pipeline("This a sentence. This is another sentence.")
```

Refer to [CoBaLD models](https://huggingface.co/CoBaLD), `ConlluTokenClassificationPipeline` implementation and huggingface [pipeline documentation](https://huggingface.co/docs/transformers/main/en/pipeline_tutorial) for details.

To inference model on pre-tokenized texts, see `predict_pretokenized.py` script.

## Citation

```bibtex
@inproceedings{baiuk2025cobald,
  title={CoBaLD Parser: Joint Morphosyntactic and Semantic Annotation},
  author={Baiuk, Ilia and Baiuk, Alexandra and Petrova, Maria},
  booktitle={Proceedings of the International Conference "Dialogue"},
  volume={I},
  year={2025}
}
```
