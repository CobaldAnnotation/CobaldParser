#!/usr/bin/env bash

MODEL_PATH=$1
INPUT_CONLLU_PATH=$(realpath $2)
OUTPUT_CONLLU_PATH=$(realpath $3)

# Preprocess data: remove range tokens (\d-\d) from input conllu file.
cd data
PREPROCESSED_INPUT_CONLLU_PATH=$(realpath _tmp_test_preprocessed.conllu)
./preprocessing.py $INPUT_CONLLU_PATH $PREPROCESSED_INPUT_CONLLU_PATH
cd ..

# Inference model against preprocessed input.
OUTPUT_RAW_CONLLU_PATH=$(realpath _tmp_predictions_raw.conllu)
allennlp predict $MODEL_PATH $PREPROCESSED_INPUT_CONLLU_PATH \
    --output-file $OUTPUT_RAW_CONLLU_PATH \
    --include-package src \
    --predictor morpho_syntax_semantic_predictor \
    --use-dataset-reader \
    --silent

# Postprocess model predictions: add range tokens from original input conllu.
cd data
./postprocessing.py $INPUT_CONLLU_PATH $OUTPUT_RAW_CONLLU_PATH $OUTPUT_CONLLU_PATH
cd ..

# Delete tmp files.
rm $PREPROCESSED_INPUT_CONLLU_PATH $OUTPUT_RAW_CONLLU_PATH
