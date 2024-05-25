#!/usr/bin/env bash

# Download dataset.conllu
git clone https://github.com/CobaldAnnotation/CobaldEng
mv CobaldEng/enhanced/train.conllu target_dataset.conllu
rm -rf CobaldEng
# Do some preprocessing and split dataset into train/validation.
./preprocessing.py target_dataset.conllu target_dataset_processed.conllu
./filter_invalid_conllu.py target_dataset_processed.conllu target_dataset_filtered.conllu
./train_val_split.py target_dataset_filtered.conllu train.conllu validation.conllu 0.8
# Cleanup
rm target_dataset_processed.conllu target_dataset_filtered.conllu
