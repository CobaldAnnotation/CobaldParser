import json
import itertools

import numpy as np
import torch
from torch import LongTensor
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    ClassLabel,
    concatenate_datasets
)

from src.lemmatize_helper import construct_lemma_rule
from cobald_parser.utils import pad_sequences


ROOT_HEAD = '0'
NULL = "#NULL"

# Sentence metadata
SENT_ID = "sent_id"
TEXT = "text"

# Fields
ID = "id"
WORD = "word"
LEMMA = "lemma"
UPOS = "upos"
XPOS = "xpos"
FEATS = "feats"
HEAD = "head"
DEPREL = "deprel"
DEPS = "deps"
MISC = "misc"
DEEPSLOT = "deepslot"
SEMCLASS = "semclass"

# Updated fields
COUNTING_MASK = "counting_mask"
LEMMA_RULE = "lemma_rule"
JOINT_FEATS = "joint_feats"
UD_ARC_FROM = "ud_arc_from"
UD_ARC_TO = "ud_arc_to"
UD_DEPREL = "ud_deprel"
EUD_ARC_FROM = "eud_arc_from"
EUD_ARC_TO = "eud_arc_to"
EUD_DEPREL = "eud_deprel"


def remove_range_tokens(sentence: dict) -> dict:
    """
    Remove range tokens from a sentence.
    """
    def is_range_id(idtag: str) -> bool:
        return '-' in idtag
    
    sentence_length = len(sentence[ID])
    return {
        key: [values[i]
              for i in range(sentence_length)
              if not is_range_id(sentence[ID][i])]
        for key, values in sentence.items()
        if values is not None and isinstance(values, list)
    }


def build_counting_mask(words: list[str]) -> np.array:
    """
    Count the number of nulls following each non-null token for a bunch of sentences.
    `counting_mask[i] = N` means i-th non-null token is followed by N nulls.
    
    FIXME: move to tests
    >>> words = ["#NULL", 'Quick', "#NULL", "#NULL", 'brown', 'fox', "#NULL"]
    >>> build_counting_mask(words)
    array([1, 2, 0, 1])
    """
    # -1 accounts for leading nulls and len(words) accounts for the trailing nulls.
    nonnull_words_idxs = [-1] + [i for i, word in enumerate(words) if word != NULL] + [len(words)]
    counting_mask = np.diff(nonnull_words_idxs) - 1
    return counting_mask


def renumerate_heads(ids: list[str], arcs_from: list[str], heads: list[str]) -> list[dict]:
    """
    Renumerate ids, so that #NULLs get integer id, e.g. [1, 1.1, 2] turns into [0, 1, 2].
    Also renumerates deps' heads, starting indexing at 0 and replacing ROOT with self-loop,
    e.g. [2, 0, 1, 2] -> [1, 1, 0, 1]. It makes parser implementation much easier.
    """
    old2new_id = {old_id: new_id for new_id, old_id in enumerate(ids)}

    arcs_to = [
        old2new_id[head]
        if head != ROOT_HEAD else token_index
        for token_index, head in zip(arcs_from, heads, strict=True)
    ]
    return arcs_to


def transform_fields(sentence: dict) -> dict:
    """
    Transform sentence fields:
     * turn words and lemmas into lemma rules,
     * merge upos, xpos and feats into "pos-feats",
     * encode ud syntax into a single 2d matrix,
     * same for e-ud syntax.
    """
    result = {}

    result[COUNTING_MASK] = build_counting_mask(sentence[WORD])

    if LEMMA in sentence:
        result[LEMMA_RULE] = [
            construct_lemma_rule(word, lemma)
            if lemma is not None else None
            for word, lemma in zip(sentence[WORD], sentence[LEMMA], strict=True)
        ]
    
    if UPOS in sentence or XPOS in sentence or FEATS in sentence:
        result[JOINT_FEATS] = [
            f"{upos}#{xpos}#{feats}"
            if (upos is not None or xpos is not None or feats is not None) else None
            for upos, xpos, feats in zip(sentence[UPOS], sentence[XPOS], sentence[FEATS], strict=True)
        ]

    # Basic syntax.
    if HEAD in sentence and DEPREL in sentence:
        ud_arcs_from, ud_heads, ud_deprels = zip(
            *[
                (token_index, str(head), rel)
                for token_index, (head, rel) in enumerate(zip(sentence[HEAD], sentence[DEPREL], strict=True))
                if head is not None
            ],
            strict=True
        )
        result[UD_ARC_FROM] = ud_arcs_from
        result[UD_ARC_TO] = renumerate_heads(sentence[ID], ud_arcs_from, ud_heads)
        result[UD_DEPREL] = ud_deprels

    # Enhanced syntax.
    if DEPS in sentence:
        eud_arcs_from, eud_heads, eud_deprels = zip(
            *[
                (token_index, head, rel)
                for token_index, deps in enumerate(sentence[DEPS])
                for head, rel in json.loads(deps).items()
                if deps is not None
            ],
            strict=True
        )
        result[EUD_ARC_FROM] = eud_arcs_from
        result[EUD_ARC_TO] = renumerate_heads(sentence[ID], eud_arcs_from, eud_heads)
        result[EUD_DEPREL] = eud_deprels

    return result


def extract_unique_labels(dataset, column_name) -> list[str]:
    """Extract unique labels from a specific column in the dataset."""
    all_labels = itertools.chain.from_iterable(dataset[column_name])
    unique_labels = set(all_labels)
    unique_labels.discard(None)
    # Ensure consistent ordering of labels
    return sorted(unique_labels)


def build_schema_with_class_labels(dataset: Dataset) -> Features:
    """Update the schema to use ClassLabel for specified columns."""

    # Updated features schema
    features = Features({
        SENT_ID: Value("string"),
        TEXT: Value("string"),
        WORD: Sequence(Value("string"))
    })

    max_null_count = max(itertools.chain.from_iterable(dataset[COUNTING_MASK]))
    features[COUNTING_MASK] = Sequence(ClassLabel(num_classes=max_null_count + 1))

    # Extract unique labels for each column that needs to be ClassLabel.
    if LEMMA_RULE in dataset.column_names:
        lemma_rule_tagset = extract_unique_labels(dataset, LEMMA_RULE)
        features[LEMMA_RULE] = Sequence(ClassLabel(names=lemma_rule_tagset))

    if JOINT_FEATS in dataset.column_names:
        joint_feats_tagset = extract_unique_labels(dataset, JOINT_FEATS)
        features[JOINT_FEATS] = Sequence(ClassLabel(names=joint_feats_tagset))

    if UD_DEPREL in dataset.column_names:
        features[UD_ARC_FROM] = Sequence(Value('int32'))
        features[UD_ARC_TO] = Sequence(Value('int32'))
        ud_deprels_tagset = extract_unique_labels(dataset, UD_DEPREL)
        features[UD_DEPREL] = Sequence(ClassLabel(names=ud_deprels_tagset))

    if EUD_DEPREL in dataset.column_names:
        features[EUD_ARC_FROM] = Sequence(Value('int32'))
        features[EUD_ARC_TO] = Sequence(Value('int32'))
        eud_deprels_tagset = extract_unique_labels(dataset, EUD_DEPREL)
        features[EUD_DEPREL] = Sequence(ClassLabel(names=eud_deprels_tagset))

    if MISC in dataset.column_names:
        misc_tagset = extract_unique_labels(dataset, MISC)
        features[MISC] = Sequence(ClassLabel(names=misc_tagset))

    if DEEPSLOT in dataset.column_names:
        deepslot_tagset = extract_unique_labels(dataset, DEEPSLOT)
        features[DEEPSLOT] = Sequence(ClassLabel(names=deepslot_tagset))

    if SEMCLASS in dataset.column_names:
        semclass_tagset = extract_unique_labels(dataset, SEMCLASS)
        features[SEMCLASS] = Sequence(ClassLabel(names=semclass_tagset))

    return features


def replace_none(example: dict, value: int) -> dict:
    """
    Replace None labels with specified value.
    """
    for name, column in example.items():
        # Skip metadata fields (they are not lists).
        if isinstance(column, list):
            example[name] = [value if item is None else item for item in column]
    return example


def preprocess(dataset_dict: DatasetDict, none_value: int = -100) -> Dataset:
    # Remove range tokens.
    dataset_dict = dataset_dict.map(remove_range_tokens)

    # Transform fields.
    dataset_column_names = {
        column
        for columns in dataset_dict.column_names.values()
        for column in columns
    }
    columns_to_remove = [ID, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS]
    dataset_dict = dataset_dict.map(
        transform_fields,
        remove_columns=[
            column
            for column in columns_to_remove
            if column in dataset_column_names
        ]
    )

    # Encode labels (str -> int).
    # FIXME: Should be a trainig schema with OOV and special handling in evaluation
    # but it makes things too complicated.
    all_data = concatenate_datasets(dataset_dict.values())
    all_schema = build_schema_with_class_labels(all_data)
    dataset_dict = dataset_dict.cast(all_schema)
    # Replace None labels with ingore_index on-the-fly.
    dataset_dict = dataset_dict.map(lambda sample: replace_none(sample, none_value))
    # Convert list labels to tensors.
    return dataset_dict.with_format("torch")


def collate_with_padding(batches: list[dict], padding_value: int = -100) -> dict:
    def gather_column(column_name: str) -> list:
        return [batch[column_name] for batch in batches]

    def stack_padded(column_name) -> LongTensor:
        return pad_sequences(gather_column(column_name), padding_value)

    def collate_syntax(arcs_from_name: str, arcs_to_name: str, deprel_name: str) -> LongTensor:
        batch_size = len(batches)
        arcs_counts = torch.tensor([len(batch[arcs_from_name]) for batch in batches])
        batch_idxs = torch.arange(batch_size).repeat_interleave(arcs_counts)
        from_idxs = torch.concat(gather_column(arcs_from_name))
        to_idxs = torch.concat(gather_column(arcs_to_name))
        deprels = torch.concat(gather_column(deprel_name))
        return torch.stack([batch_idxs, from_idxs, to_idxs, deprels], dim=1)

    def maybe_none(labels: LongTensor) -> LongTensor | None:
        return None if labels.max() == padding_value or labels.numel() == 0 else labels
    
    result = {
        "words": gather_column(WORD),
        "sent_ids": gather_column(SENT_ID),
        "texts": gather_column(TEXT)
    }

    counting_masks_batched = stack_padded(COUNTING_MASK)
    result["counting_masks"] = maybe_none(counting_masks_batched)

    columns = {column for batch in batches for column in batch}
    if LEMMA_RULE in columns:
        lemma_rules_batched = stack_padded(LEMMA_RULE)
        result["lemma_rules"] = maybe_none(lemma_rules_batched)

    if JOINT_FEATS in columns:
        joint_feats_batched = stack_padded(JOINT_FEATS)
        result["joint_feats"] = maybe_none(joint_feats_batched)

    if UD_DEPREL in columns:
        deps_ud_batched = collate_syntax(UD_ARC_FROM, UD_ARC_TO, UD_DEPREL)
        result["deps_ud"] = maybe_none(deps_ud_batched)

    if EUD_DEPREL in columns:
        deps_eud_batched = collate_syntax(EUD_ARC_FROM, EUD_ARC_TO, EUD_DEPREL)
        result["deps_eud"] = maybe_none(deps_eud_batched)

    if MISC in columns:
        miscs_batched = stack_padded(MISC)
        result["miscs"] = maybe_none(miscs_batched)

    if DEEPSLOT in columns:
        deepslots_batched = stack_padded(DEEPSLOT)
        result["deepslots"] = maybe_none(deepslots_batched)

    if SEMCLASS in columns:
        semclasses_batched = stack_padded(SEMCLASS)
        result["semclasses"] = maybe_none(semclasses_batched)

    return result