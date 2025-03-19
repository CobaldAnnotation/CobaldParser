import json
import itertools
from datasets import Dataset, Features, Sequence, Value, ClassLabel

from torch import Tensor

from src.lemmatize_helper import construct_lemma_rule, reconstruct_lemma
from src.utils import pad_sequences, pad_matrices, IGNORE_INDEX


def renumerate_syntax(ids: list[str], heads: list[int], deps: list[str]):
    """
    Renumerate ids, so that #NULLs get integer id, e.g. [1, 1.1, 2] turns into [0, 1, 2].
    Also renumerates heads, starting indexing at 0 and replacing ROOT with self-loop,
    e.g. [2, 0, 1, 2] -> [1, 1, 0, 1]. It makes parser implementation much easier.
    Deps are renumerated similarly.
    """
    ROOT = '0'

    old2new_id = {None: None}
    old2new_id |= {old_id: new_id for new_id, old_id in enumerate(ids)}

    # Renumareate heads and replace ROOT with self-loop.
    new_heads = []
    for token_index, head in enumerate(heads):
        # cast to string to reuse logic for deps
        head = str(head) if head is not None else None
        assert head != token_index
        if head != ROOT:
            new_heads.append(old2new_id[head])
        else:
            new_heads.append(token_index)

    # Same for deps.
    new_deps = []
    for token_index, token_deps in enumerate(deps):
        if token_deps is None:
            new_deps.append(None)
            continue
        dep_dict = {}
        for head, rel in json.loads(token_deps).items():
            assert head != token_index
            new_head = old2new_id[head] if head != ROOT else token_index
            dep_dict[new_head] = rel
        new_deps.append(dep_dict)

    return new_heads, new_deps


def remove_range_tokens(sentence: dict[str, list]) -> dict[str, list]:
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


def transform_fields(sentence: dict[str, list]) -> dict[str, list]:
    """
    Transform sentence fields:
     * turn words and lemmas into lemma rules,
     * merge upos, xpos and feats into "pos-feats",
     * encode ud syntax into a single 2d matrix,
     * same for e-ud syntax.
    """

    lemma_rules = [
        construct_lemma_rule(word, lemma) if lemma is not None else None
        for word, lemma in zip(sentence["words"], sentence["lemmas"], strict=True)
    ]

    joint_pos_feats = [
        f"{upos}#{xpos}#{feats}"
        if (upos is not None or xpos is not None or feats is not None) else None
        for upos, xpos, feats in zip(sentence["upos"], sentence["xpos"], sentence["feats"], strict=True)
    ]

    new_heads, new_deps = renumerate_syntax(sentence["ids"], sentence["heads"], sentence["deps"])

    seq_len = len(sentence["ids"])
    deps_matrix_ud = [[None] * seq_len for _ in range(seq_len)]
    for token_index, (head, relation) in enumerate(zip(new_heads, sentence["deprels"], strict=True)):
        if head is not None:
            deps_matrix_ud[token_index][head] = relation

    deps_matrix_eud = [[None] * seq_len for _ in range(seq_len)]
    for token_index, token_deps in enumerate(new_deps):
        if token_deps is not None:
            for head, relation in token_deps.items():
                deps_matrix_eud[token_index][head] = relation

    return {
        "lemma_rule_labels": lemma_rules,
        "joint_pos_feats_labels": joint_pos_feats,
        "deps_ud_labels": deps_matrix_ud,
        "deps_eud_labels": deps_matrix_eud,
        "misc_labels": sentence["miscs"],
        "deepslot_labels": sentence["deepslots"],
        "semclass_labels": sentence["semclasses"],
    }


def extract_unique_labels(dataset, column_name, is_matrix=False) -> list[str]:
    """Extract unique labels from a specific column in the dataset."""
    if is_matrix:
        all_labels = [
            value
            for matrices in dataset[column_name]
            for matrix in matrices
            for value in matrix
        ]
    else:
        all_labels = itertools.chain.from_iterable(dataset[column_name])

    unique_labels = set(all_labels)
    unique_labels.discard(None)
    # Ensure consistent ordering of labels
    return sorted(unique_labels)


def update_schema_with_class_labels(dataset: Dataset) -> Features:
    """Update the schema to use ClassLabel for specified columns."""

    # Extract unique labels for each column that needs to be ClassLabel.
    lemma_rule_tagset = extract_unique_labels(dataset, "lemma_rule_labels")
    joint_pos_feats_tagset = extract_unique_labels(dataset, "joint_pos_feats_labels")
    deps_ud_tagset = extract_unique_labels(dataset, "deps_ud_labels", is_matrix=True)
    deps_eud_tagset = extract_unique_labels(dataset, "deps_eud_labels", is_matrix=True)
    misc_tagset = extract_unique_labels(dataset, "misc_labels")
    deepslot_tagset = extract_unique_labels(dataset, "deepslot_labels")
    semclass_tagset = extract_unique_labels(dataset, "semclass_labels")

    # Define updated features schema
    features = Features({
        "words": Sequence(Value("string")),
        "lemma_rule_labels": Sequence(ClassLabel(names=lemma_rule_tagset)),
        "joint_pos_feats_labels": Sequence(ClassLabel(names=joint_pos_feats_tagset)),
        "deps_ud_labels": Sequence(Sequence(ClassLabel(names=deps_ud_tagset))),
        "deps_eud_labels": Sequence(Sequence(ClassLabel(names=deps_eud_tagset))),
        "misc_labels": Sequence(ClassLabel(names=misc_tagset)),
        "deepslot_labels": Sequence(ClassLabel(names=deepslot_tagset)),
        "semclass_labels": Sequence(ClassLabel(names=semclass_tagset)),
        "sent_id": Value("string"),
        "text": Value("string")
    })
    return features


def replace_none_with_ignore_index(example: dict) -> dict:
    """
    Replace None labels with ignore_index.
    """
    for column in example.values():
        if not isinstance(column, list):
            # Skip metadata fields.
            continue
        for i in range(len(column)):
            if isinstance(column[i], list):
                # Syntactic fields are nested lists.
                column[i] = [IGNORE_INDEX if label is None else label for label in column[i]]
            else:
                # Morphological and semantic fields are simple lists.
                column[i] = IGNORE_INDEX if column[i] is None else column[i]
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
            'deps',
            'miscs',
            'deepslots',
            'semclasses'
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


def collate_with_ignore_index(batches: list[dict[str, list | Tensor]]) -> dict[str, list | Tensor | None]:
    def stack_list_column(column):
        return [batch[column] for batch in batches]

    def pad_sequence_column(column):
        return pad_sequences([batch[column] for batch in batches], IGNORE_INDEX)

    def pad_matrix_column(column):
        return pad_matrices([batch[column] for batch in batches], IGNORE_INDEX)

    def return_non_empty(labels):
        return labels if labels.max() != IGNORE_INDEX else None

    lemma_rules = pad_sequence_column('lemma_rule_labels')
    joint_pos_feats = pad_sequence_column('joint_pos_feats_labels')
    deps_ud = pad_matrix_column('deps_ud_labels')
    deps_eud = pad_matrix_column('deps_eud_labels')
    miscs = pad_sequence_column('misc_labels')
    deepslots = pad_sequence_column('deepslot_labels')
    semclasses = pad_sequence_column('semclass_labels')

    return {
        "words": stack_list_column('words'),
        "lemma_rule_labels": return_non_empty(lemma_rules),
        "joint_pos_feats_labels": return_non_empty(joint_pos_feats),
        "deps_ud_labels": return_non_empty(deps_ud),
        "deps_eud_labels": return_non_empty(deps_eud),
        "misc_labels": return_non_empty(miscs),
        "deepslot_labels": return_non_empty(deepslots),
        "semclass_labels": return_non_empty(semclasses),
        "sent_id": stack_list_column('sent_id'),
        "text": stack_list_column('text')
    }


################################################################################


def restore_ids(sentence: list[str]) -> list[str]:
    ids = []

    current_id = 0
    current_null_count = 0
    for word in sentence:
        if word == "#NULL":
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
    joint_pos_feats: list[str],
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
    tokens_labels = zip(ids, words, lemma_rules, joint_pos_feats, deps_ud, deps_eud, miscs, deepslots, semclasses)
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
        token["misc"] = misc if word != "#NULL" else 'ellipsis'
        token["deepslot"] = deepslot
        token["semclass"] = semclass
        # Add token to a result sentence.
        tokens.append(token)

    return tokens
