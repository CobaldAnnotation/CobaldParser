import itertools
from typing import Iterable

import sys
sys.path.append("..")
from common.token import Token
from common.sentence import Sentence
from common.parse_conllu import parse_conllu_incr

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from datasets import Dataset, DatasetDict, Features, Sequence, Value, ClassLabel

from lemmatize_helper import predict_lemma_rule
from utils import dict_from_str, pad_matrices


def convert_sentence_to_dict(sentence: Sentence) -> dict:
    return {
        "words": sentence.words,
        "lemmas": sentence.lemmas,
        "upos": sentence.upos,
        "xpos": sentence.xpos,
        "feats": sentence.feats,
        "heads": sentence.heads,
        "deprels": sentence.deprels,
        "deps": sentence.deps,
        "miscs": sentence.miscs,
        "deepslots": sentence.semslots,
        "semclasses": sentence.semclasses,
        "metadata": sentence.metadata
    }


def convert_conllu_to_hf(file_path: str) -> Iterable[dict]:
    with open(file_path, "r") as file:
        for sentence in parse_conllu_incr(file):
            yield convert_sentence_to_dict(sentence)


def build_raw_dataset(file_paths: dict[str, str]) -> DatasetDict:
    features = Features({
        "words": Sequence(Value("string")),
        "lemmas": Sequence(Value("string")),
        "upos": Sequence(Value("string")),
        "xpos": Sequence(Value("string")),
        "feats": Sequence(Value("string")),
        "heads": Sequence(Value("int32")),
        "deprels": Sequence(Value("string")),
        "deps": Sequence(Value("string")),
        "miscs": Sequence(Value("string")),
        "deepslots": Sequence(Value("string")),
        "semclasses": Sequence(Value("string")),
        "metadata": Value("string")
    })

    splits = {}
    for split_name, file_path in file_paths.items():
        splits[split_name] = Dataset.from_generator(
            convert_conllu_to_hf,
            gen_kwargs={"file_path": file_path},
            features=features
        )
    return DatasetDict(splits)


def transform_columns(batch: dict[str, list]) -> dict[str, list]:
    """Transform raw string labels to TODO."""

    words = batch["words"]
    lemmas = batch["lemmas"]
    upos = batch["upos"]
    xpos = batch["xpos"]
    feats = batch["feats"]
    heads = batch["heads"]
    deprels = batch["deprels"]
    deps = batch["deps"]

    lemma_rules: list[str] = None
    if lemmas is not None:
        lemma_rules = [
            str(predict_lemma_rule(word if word is not None else '', lemma if lemma is not None else ''))
            for word, lemma in zip(words, lemmas, strict=True)
        ]

    joint_pos_feats: list[str] = None
    if upos is not None and xpos is not None and feats is not None:
        joint_feats = [
            '|'.join([f"{k}={v}" for k, v in dict_from_str(feat).items()]) if 0 < len(dict_from_str(feat)) else '_'
            for feat in feats
        ]
        joint_pos_feats = [
            f"{token_upos}#{token_xpos}#{token_joint_feats}"
            for token_upos, token_xpos, token_joint_feats in zip(upos, xpos, joint_feats, strict=True)
        ]

    sequence_length = len(words)

    deps_matrix_ud = None
    if heads is not None and deprels is not None:
        deps_matrix_ud = [[''] * sequence_length for _ in range(sequence_length)]
        for index, (head, relation) in enumerate(zip(heads, deprels, strict=True)):
            # Skip nulls.
            if head == -1:
                continue
            assert 0 <= head
            # Hack: start indexing at 0 and replace ROOT with self-loop.
            # It makes parser implementation much easier.
            if head == 0:
                # Replace ROOT with self-loop.
                head = index
            else:
                # If not ROOT, shift token left.
                head -= 1
                assert head != index, f"head = {head + 1} must not be equal to index = {index + 1}"
            deps_matrix_ud[index][head] = relation

    deps_matrix_eud = None
    if deps is not None:
        deps_matrix_eud = [[''] * sequence_length for _ in range(sequence_length)]
        for index, dep in enumerate(deps):
            dep = dict_from_str(dep) # Convert string representation of dict to a dict.
            assert 0 < len(dep), f"Deps must not be empty"
            for head, relation in dep.items():
                assert 0 <= head
                # Hack: start indexing at 0 and replace ROOT with self-loop.
                # It makes parser implementation much easier.
                if head == 0:
                    # Replace ROOT with self-loop.
                    head = index
                else:
                    # If not ROOT, shift token left.
                    head -= 1
                    assert head != index, f"head = {head + 1} must not be equal to index = {index + 1}"
                deps_matrix_eud[index][head] = relation

    return {
        "lemma_rules": lemma_rules,
        "joint_pos_feats": joint_pos_feats,
        "deps_ud": deps_matrix_ud,
        "deps_eud": deps_matrix_eud
    }


def update_schema_with_class_labels(dataset_dict: DatasetDict) -> Features:
    """Convert string features to integers (ClassLabel with )."""

    def extract_unique_labels(dataset, column_name, is_matrix=False) -> list[str]:
        """Extract unique labels from a specified column in the dataset."""
        if is_matrix:
            all_labels = [value for matrices in dataset[column_name] for matrix in matrices for value in matrix]
        else:
            all_labels = itertools.chain.from_iterable(dataset[column_name])
        return sorted(set(all_labels)) # Ensure consistent ordering of labels

    # Extract labels from train dataset only, since all the labels must be present in training data.
    train_dataset = dataset_dict['train']

    # Extract unique labels for each column that needs to be ClassLabel.
    lemma_rule_labels = extract_unique_labels(train_dataset, "lemma_rules")
    joint_pos_feats_labels = extract_unique_labels(train_dataset, "joint_pos_feats")
    deps_ud_labels = extract_unique_labels(train_dataset, "deps_ud", is_matrix=True)
    deps_eud_labels = extract_unique_labels(train_dataset, "deps_eud", is_matrix=True)
    misc_labels = extract_unique_labels(train_dataset, "miscs")
    deepslot_labels = extract_unique_labels(train_dataset, "deepslots")
    semclass_labels = extract_unique_labels(train_dataset, "semclasses")

    # Define updated features schema.
    features = Features({
        "words": Sequence(Value("string")),
        "lemma_rules": Sequence(ClassLabel(names=lemma_rule_labels), ),
        "joint_pos_feats": Sequence(ClassLabel(names=joint_pos_feats_labels)),
        "deps_ud": Sequence(Sequence(ClassLabel(names=deps_ud_labels))),
        "deps_eud": Sequence(Sequence(ClassLabel(names=deps_eud_labels))),
        "miscs": Sequence(ClassLabel(names=misc_labels)),
        "deepslots": Sequence(ClassLabel(names=deepslot_labels)),
        "semclasses": Sequence(ClassLabel(names=semclass_labels)),
        "metadata": Value("string")
    })
    return features


def collate_fn(batches: list[dict[str, list | Tensor]]) -> dict[str, list | Tensor]:
    padding_value = -100
    stack_list_column = lambda column: [batch[column] for batch in batches]
    pad_sequence_column = lambda column: pad_sequence(
        [batch[column] for batch in batches],
        padding_value=padding_value,
        batch_first=True
    )
    pad_matrix_column = lambda column: pad_matrices(
        [batch[column] for batch in batches],
        padding_value=padding_value
    )
    return {
        "words": stack_list_column('words'),
        "lemma_rules": pad_sequence_column('lemma_rules'),
        "joint_pos_feats": pad_sequence_column('joint_pos_feats'),
        "deps_ud": pad_matrix_column('deps_ud'),
        "deps_eud": pad_matrix_column('deps_eud'),
        "miscs": pad_sequence_column('miscs'),
        "deepslots": pad_sequence_column('deepslots'),
        "semclasses": pad_sequence_column('semclasses'),
        "metadata": stack_list_column('metadata')
    }


# Sample
# file_paths = {
#     "train": "../data/train.conllu",
#     "validation": "../data/validation.conllu",
#     "test": "../data/test_clean.conllu",
# }
# dataset = build_raw_dataset(file_paths)
# dataset = dataset.map(preprocess, remove_columns=['lemmas', 'upos', 'xpos', 'feats', 'heads', 'deprels', 'deps'])
# 
# class_features = update_schema_with_class_labels(dataset)
# dataset = dataset.cast(class_features)
# 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset = dataset.with_format("torch", device=device)
# 
# dataloader = DataLoader(dataset['train'], batch_size=4, collate_fn=collate_fn)


# TODO: validate dataset
