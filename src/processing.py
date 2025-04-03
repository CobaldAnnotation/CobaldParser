import json
import itertools
from datasets import Dataset, Features, Sequence, Value, ClassLabel

import numpy as np

import torch
from torch import LongTensor

from src.lemmatize_helper import construct_lemma_rule, reconstruct_lemma
from src.utils import pad_sequences, IGNORE_INDEX


ROOT = '0'
NULL = "#NULL"


def remove_range_tokens(sentence: dict) -> dict:
    """
    Remove range tokens from a sentence.
    """
    def is_range_id(idtag: str) -> bool:
        return '-' in idtag
    
    sentence_length = len(sentence['ids'])
    return {
        key: [values[i]
              for i in range(sentence_length)
              if not is_range_id(sentence['ids'][i])]
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


def renumerate_heads(ids: list[str], heads: list[str]) -> list[dict]:
    """
    Renumerate ids, so that #NULLs get integer id, e.g. [1, 1.1, 2] turns into [0, 1, 2].
    Also renumerates deps' heads, starting indexing at 0 and replacing ROOT with self-loop,
    e.g. [2, 0, 1, 2] -> [1, 1, 0, 1]. It makes parser implementation much easier.
    """
    old2new_id = {old_id: new_id for new_id, old_id in enumerate(ids)}

    heads_renumerated = [
        old2new_id[head]
        if head != ROOT else token_index
        for token_index, head in enumerate(heads)
    ]
    return heads_renumerated


def transform_fields(sentence: dict) -> dict:
    """
    Transform sentence fields:
     * turn words and lemmas into lemma rules,
     * merge upos, xpos and feats into "pos-feats",
     * encode ud syntax into a single 2d matrix,
     * same for e-ud syntax.
    """

    counting_mask = build_counting_mask(sentence["words"])

    lemma_rules = [
        construct_lemma_rule(word, lemma)
        if lemma is not None else None
        for word, lemma in zip(sentence["words"], sentence["lemmas"], strict=True)
    ]
    morph_feats = [
        f"{upos}#{xpos}#{feats}"
        if (upos is not None or xpos is not None or feats is not None) else None
        for upos, xpos, feats in zip(sentence["upos"], sentence["xpos"], sentence["feats"], strict=True)
    ]
    # Basic syntax.
    ud_arcs_from, ud_heads, ud_deprels = zip(*[
        (token_index, str(head), rel)
        for token_index, (head, rel) in enumerate(zip(sentence["heads"], sentence["deprels"], strict=True))
        if head is not None
    ])
    ud_arcs_to = renumerate_heads(sentence["ids"], ud_heads)
    # Enhanced syntax.
    eud_arcs_from, eud_heads, eud_deprels = zip(*[
        (token_index, head, rel)
        for token_index, deps in enumerate(sentence["deps"])
        for head, rel in json.loads(deps).items()
        if deps is not None
    ])
    eud_arcs_to = renumerate_heads(sentence["ids"], eud_heads)

    return {
        "counting_mask": counting_mask,
        "lemma_rules": lemma_rules,
        "morph_feats": morph_feats,
        "ud_arcs_from": ud_arcs_from,
        "ud_arcs_to": ud_arcs_to,
        "ud_deprels": ud_deprels,
        "eud_arcs_from": eud_arcs_from,
        "eud_arcs_to": eud_arcs_to,
        "eud_deprels": eud_deprels
    }


def extract_unique_labels(dataset, column_name) -> list[str]:
    """Extract unique labels from a specific column in the dataset."""
    all_labels = itertools.chain.from_iterable(dataset[column_name])
    unique_labels = set(all_labels)
    unique_labels.discard(None)
    # Ensure consistent ordering of labels
    return sorted(unique_labels)


def update_schema_with_class_labels(dataset: Dataset) -> Features:
    """Update the schema to use ClassLabel for specified columns."""

    max_null_count = max(itertools.chain.from_iterable(dataset["counting_mask"]))
    # Extract unique labels for each column that needs to be ClassLabel.
    lemma_rule_tagset = extract_unique_labels(dataset, "lemma_rules")
    morph_feats_tagset = extract_unique_labels(dataset, "morph_feats")
    ud_deprels_tagset = extract_unique_labels(dataset, "ud_deprels")
    eud_deprels_tagset = extract_unique_labels(dataset, "eud_deprels")
    misc_tagset = extract_unique_labels(dataset, "miscs")
    deepslot_tagset = extract_unique_labels(dataset, "deepslots")
    semclass_tagset = extract_unique_labels(dataset, "semclasses")

    # Define updated features schema
    features = Features({
        "words": Sequence(Value("string")),
        "counting_mask": Sequence(ClassLabel(num_classes=max_null_count + 1)),
        "lemma_rules": Sequence(ClassLabel(names=lemma_rule_tagset)),
        "morph_feats": Sequence(ClassLabel(names=morph_feats_tagset)),
        "ud_arcs_from": Sequence(Value('int32')),
        "ud_arcs_to": Sequence(Value('int32')),
        "ud_deprels": Sequence(ClassLabel(names=ud_deprels_tagset)),
        "eud_arcs_from": Sequence(Value('int32')),
        "eud_arcs_to": Sequence(Value('int32')),
        "eud_deprels": Sequence(ClassLabel(names=eud_deprels_tagset)),
        "miscs": Sequence(ClassLabel(names=misc_tagset)),
        "deepslots": Sequence(ClassLabel(names=deepslot_tagset)),
        "semclasses": Sequence(ClassLabel(names=semclass_tagset)),
        "sent_id": Value("string"),
        "text": Value("string")
    })
    return features


def replace_none_with_ignore_index(example: dict) -> dict:
    """
    Replace None labels with IGNORE_INDEX.
    """
    for name, column in example.items():
        # Skip metadata fields (they are not lists).
        if isinstance(column, list):
            example[name] = [IGNORE_INDEX if val is None else val for val in column]
    return example


def preprocess(dataset: Dataset) -> Dataset:
    # Remove range tokens.
    dataset = dataset.map(remove_range_tokens, batched=False)
    # Transform fields.
    dataset = dataset.map(
        transform_fields,
        remove_columns=[
            'ids',
            'lemmas',
            'upos',
            'xpos',
            'feats',
            'heads',
            'deprels',
            'deps'
        ],
        batched=False
    )
    # Encode labels (str -> int).
    class_features = update_schema_with_class_labels(dataset)
    dataset = dataset.cast(class_features)
    # Replace None labels with ingore_index.
    dataset = dataset.map(replace_none_with_ignore_index, batched=False)
    # Convert list labels to tensors.
    dataset = dataset.with_format("torch")
    return dataset


def maybe_none(labels: LongTensor) -> LongTensor | None:
    return None if labels.max() == IGNORE_INDEX or labels.numel() == 0 else labels

def collate_with_ignore_index(batches: list[dict]) -> dict:
    def gather_column(column_name: str) -> list:
        return [batch[column_name] for batch in batches]

    def gather_tensor_column(column_name: str) -> list:
        return [LongTensor(batch[column_name]) for batch in batches]

    def stack_padded(column_name) -> LongTensor:
        return pad_sequences(gather_tensor_column(column_name), IGNORE_INDEX)

    def collate_syntax(arcs_from_name: str, arcs_to_name: str, deprel_name: str) -> LongTensor:
        batch_size = len(batches)
        arcs_counts = LongTensor([len(batch[arcs_from_name]) for batch in batches])
        batch_idxs = torch.arange(batch_size).repeat_interleave(arcs_counts)
        from_idxs = torch.concat(gather_tensor_column(arcs_from_name))
        to_idxs = torch.concat(gather_tensor_column(arcs_to_name))
        deprels = torch.concat(gather_tensor_column(deprel_name))
        return torch.stack([batch_idxs, from_idxs, to_idxs, deprels], dim=1)

    counting_masks_batched = stack_padded('counting_mask')
    lemma_rules_batched = stack_padded('lemma_rules')
    morph_feats_batched = stack_padded('morph_feats')
    syntax_ud_batched = collate_syntax('ud_arcs_from', 'ud_arcs_to', 'ud_deprels')
    syntax_eud_batched = collate_syntax('eud_arcs_from', 'eud_arcs_to', 'eud_deprels')
    miscs_batched = stack_padded('miscs')
    deepslots_batched = stack_padded('deepslots')
    semclasses_batched = stack_padded('semclasses')

    return {
        "words": gather_column('words'),
        "counting_mask": maybe_none(counting_masks_batched),
        "lemma_rules": maybe_none(lemma_rules_batched),
        "morph_feats": maybe_none(morph_feats_batched),
        "syntax_ud": maybe_none(syntax_ud_batched),
        "syntax_eud": maybe_none(syntax_eud_batched),
        "miscs": maybe_none(miscs_batched),
        "deepslots": maybe_none(deepslots_batched),
        "semclasses": maybe_none(semclasses_batched),
        "sent_id": gather_column('sent_id'),
        "text": gather_column('text')
    }


################################################################################


def restore_ids(sentence: list[str]) -> list[str]:
    ids = []

    current_id = 0
    current_null_count = 0
    for word in sentence:
        if word == NULL:
            current_null_count += 1
            ids.append(f"{current_id}.{current_null_count}")
        else:
            current_id += 1
            current_null_count = 0
            ids.append(f"{current_id}")
    return ids


def postprocess(
    words: list[str],
    lemma_rules: list[str],
    morph_feats: list[str],
    deps_ud: list[list[str]],
    deps_eud: list[list[str]],
    miscs: list[str],
    deepslots: list[str],
    semclasses: list[str]
) -> list:

    ids = restore_ids(words)
    # Renumerate heads back, e.g. [0, 1, 2] into [1, 1.1, 2].
    # Luckily, we already have this mapping stored in `ids`.
    new2old_id = {i: id for i, id in enumerate(ids)}

    tokens = []
    tokens_labels = zip(ids, words, lemma_rules, morph_feats, deps_ud, deps_eud, miscs, deepslots, semclasses)
    for i, token_labels in enumerate(tokens_labels):
        idtag, word, lemma_rule, joint_pos_feat, word_deps_ud, word_deps_eud, misc, deepslot, semclass = token_labels
        token = {"id": idtag, "word": word}
        token["lemma"] = reconstruct_lemma(word, lemma_rule)
        token["upos"], token["xpos"], token["feats"] = joint_pos_feat.split('#')

        # Syntax.
        edge_to = i # alias
        collect_heads_and_deps = lambda word_relations: {
            (new2old_id[edge_from] if edge_from != edge_to else 0): deprel
            for edge_from, deprel in enumerate(word_relations) if deprel != None
        }

        heads_and_deprels: dict[str, str] = collect_heads_and_deps(word_deps_ud)
        assert len(heads_and_deprels) <= 1, f"Token must have no more than one basic syntax head"
        if len(heads_and_deprels) == 1:
            token["head"], token["deprel"] = heads_and_deprels.popitem()
        else:
            token["head"], token["deprel"] = None, '_'

        # Enhanced syntax.
        token_deps_dict: dict[str, str] = collect_heads_and_deps(word_deps_eud)
        token["deps"] = '|'.join([f"{head}:{rel}" for head, rel in token_deps_dict.items()])

        # Force set `ellipsis` misc to nulls.
        token["misc"] = misc if word != NULL else 'ellipsis'
        token["deepslot"] = deepslot
        token["semclass"] = semclass
        # Add token to a result sentence.
        tokens.append(token)

    return tokens
