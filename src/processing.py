import json
import itertools
from datasets import Dataset, Features, Sequence, Value, ClassLabel

from torch import Tensor
import numpy as np

from src.lemmatize_helper import construct_lemma_rule, reconstruct_lemma
from src.utils import pad_sequences, pad_matrices, IGNORE_INDEX


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


def renumerate_syntax(ids: list[str], sentence_deps: list[dict]) -> list[dict]:
    """
    Renumerate ids, so that #NULLs get integer id, e.g. [1, 1.1, 2] turns into [0, 1, 2].
    Also renumerates deps' heads, starting indexing at 0 and replacing ROOT with self-loop,
    e.g. [2, 0, 1, 2] -> [1, 1, 0, 1]. It makes parser implementation much easier.
    """
    old2new_id = {old_id: new_id for new_id, old_id in enumerate(ids)}

    new_sentence_deps = []
    for token_index, token_deps in enumerate(sentence_deps):
        # Skip empty labels.
        if token_deps is None:
            new_sentence_deps.append(None)
            continue
        # Renumerate token's deps.
        new_token_deps = {}
        for head, relation in token_deps.items():
            assert head != token_index
            new_head = old2new_id[head] if head != ROOT else token_index
            new_token_deps[new_head] = relation
        new_sentence_deps.append(new_token_deps)

    return new_sentence_deps


def build_syntax_matrix(sentence_deps: list[dict]) -> list[list]:
    """
    Builds a syntax matrix from a list of tokens dependencies.
    Args:
        sentence_deps (list[dict]): A list where each element is a dictionary representing
                                    the dependencies of a token in the sentence. The keys
                                    of the dictionary are the indices of the head tokens,
                                    and the values are the syntactic relations.
    Returns:
        list[list]: A 2D list (matrix) where the element at [i][j] represents the syntactic
                    relation between the token at index i and the token at index j. If there
                    is no relation, the element is None.
    """
    seq_len = len(sentence_deps)

    syntax_matrix = [[None] * seq_len for _ in range(seq_len)]

    for token_index, token_deps in enumerate(sentence_deps):
        if token_deps is not None:
            for head, relation in token_deps.items():
                syntax_matrix[token_index][head] = relation

    return syntax_matrix


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
    nonnull_words_idxs = np.array(nonnull_words_idxs)
    counting_mask = np.diff(nonnull_words_idxs) - 1
    return counting_mask


def transform_fields(sentence: dict) -> dict:
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

    morph_feats = [
        f"{upos}#{xpos}#{feats}"
        if (upos is not None or xpos is not None or feats is not None) else None
        for upos, xpos, feats in zip(sentence["upos"], sentence["xpos"], sentence["feats"], strict=True)
    ]

    # Standardize UD and E-UD syntax.
    deps_ud = [
        {str(head): deprel}
        if head is not None else None
        for head, deprel in zip(sentence["heads"], sentence["deprels"], strict=True)
    ]
    deps_eud = [json.loads(token_deps) for token_deps in sentence["deps"]]
    new_deps_ud = renumerate_syntax(sentence["ids"], deps_ud)
    new_deps_eud = renumerate_syntax(sentence["ids"], deps_eud)
    syntax_matrix_ud = build_syntax_matrix(new_deps_ud)
    syntax_matrix_eud = build_syntax_matrix(new_deps_eud)

    counting_mask = build_counting_mask(sentence["words"])

    return {
        "lemma_rules": lemma_rules,
        "morph_feats": morph_feats,
        "syntax_ud": syntax_matrix_ud,
        "syntax_eud": syntax_matrix_eud,
        "counting_mask": counting_mask
    }


def extract_unique_labels(dataset, column_name, is_matrix = False) -> list[str]:
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
    lemma_rule_tagset = extract_unique_labels(dataset, "lemma_rules")
    morph_feats_tagset = extract_unique_labels(dataset, "morph_feats")
    ud_deprels_tagset = extract_unique_labels(dataset, "syntax_ud", is_matrix=True)
    eud_deprels_tagset = extract_unique_labels(dataset, "syntax_eud", is_matrix=True)
    misc_tagset = extract_unique_labels(dataset, "miscs")
    deepslot_tagset = extract_unique_labels(dataset, "deepslots")
    semclass_tagset = extract_unique_labels(dataset, "semclasses")
    max_null_count = max(itertools.chain.from_iterable(dataset["counting_mask"]))

    # Define updated features schema
    features = Features({
        "words": Sequence(Value("string")),
        "lemma_rules": Sequence(ClassLabel(names=lemma_rule_tagset)),
        "morph_feats": Sequence(ClassLabel(names=morph_feats_tagset)),
        "syntax_ud": Sequence(Sequence(ClassLabel(names=ud_deprels_tagset))),
        "syntax_eud": Sequence(Sequence(ClassLabel(names=eud_deprels_tagset))),
        "miscs": Sequence(ClassLabel(names=misc_tagset)),
        "deepslots": Sequence(ClassLabel(names=deepslot_tagset)),
        "semclasses": Sequence(ClassLabel(names=semclass_tagset)),
        "counting_mask": Sequence(ClassLabel(num_classes=max_null_count + 1)),
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


def collate_with_ignore_index(batches: list[dict]) -> dict:
    def stack_list_column(column):
        return [batch[column] for batch in batches]

    def pad_sequence_column(column):
        return pad_sequences([batch[column] for batch in batches], IGNORE_INDEX)

    def pad_matrix_column(column):
        return pad_matrices([batch[column] for batch in batches], IGNORE_INDEX)

    def return_non_empty(labels):
        return labels if labels.max() != IGNORE_INDEX else None

    lemma_rules_batched = pad_sequence_column('lemma_rules')
    morph_feats_batched = pad_sequence_column('morph_feats')
    syntax_ud_batched = pad_matrix_column('syntax_ud')
    syntax_eud_batched = pad_matrix_column('syntax_eud')
    miscs_batched = pad_sequence_column('miscs')
    deepslots_batched = pad_sequence_column('deepslots')
    semclasses_batched = pad_sequence_column('semclasses')
    counting_masks_batched = pad_sequence_column('counting_mask')

    return {
        "words": stack_list_column('words'),
        "lemma_rules": return_non_empty(lemma_rules_batched),
        "morph_feats": return_non_empty(morph_feats_batched),
        "syntax_ud": return_non_empty(syntax_ud_batched),
        "syntax_eud": return_non_empty(syntax_eud_batched),
        "miscs": return_non_empty(miscs_batched),
        "deepslots": return_non_empty(deepslots_batched),
        "semclasses": return_non_empty(semclasses_batched),
        "counting_mask": return_non_empty(counting_masks_batched),
        "sent_id": stack_list_column('sent_id'),
        "text": stack_list_column('text')
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
