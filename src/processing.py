from lemmatize_helper import construct_lemma_rule, reconstruct_lemma


NO_ARC_LABEL = ""

# IDs.

def _restore_ids(sentence: list[str]) -> list[str]:
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

# Lemma.

def _build_lemma_rules(words: list[str], lemmas: list[str]) -> list[str]:
    return [construct_lemma_rule(word, lemma) for word, lemma in zip(words, lemmas, strict=True)]

def _restore_lemmas(words: list[str], lemma_rules: list[str]) -> list[str]:
    return [reconstruct_lemma(word, lemma_rule) for word, lemma_rule in zip(words, lemma_rules)]

# Morphology.

def _join_pos_and_feats(upos: list[str], xpos: list[str], feats: list[str]) -> list[str]:
    return ['#'.join(morphology) for morphology in zip(upos, xpos, feats, strict=True)]

def _split_pos_and_feats(joint_pos_feats: list[str]) -> tuple[list[str], list[str], list[str]]:
    # Unzip list of tuples (xpos, upos, feats) to a tuple of lists.
    return zip(*[joint_pos_feat.split('#') for joint_pos_feat in joint_pos_feats])

# Basic syntax.

def _renumerate_heads(old2new_id: dict[str, int], heads: list[str]) -> list[int]:
    return [old2new_id[head] for head in heads]

def _build_syntax_matrix_ud(heads: list[int], deprels: list[str]) -> list[list[str]]:
    sequence_length = len(heads)

    matrix = [[NO_ARC_LABEL] * sequence_length for _ in range(sequence_length)]
    for index, (head, relation) in enumerate(zip(heads, deprels, strict=True)):
        # Skip nulls.
        if head == -1:
            continue
        assert 0 <= head
        # Trick: start indexing at 0 and replace ROOT with self-loop.
        # It makes parser implementation a bit easier.
        if head == 0:
            # Replace ROOT with self-loop.
            head = index
        else:
            # If not ROOT, shift token left.
            head -= 1
            assert head != index, f"head = {head + 1} must not be equal to index = {index + 1}"
        matrix[index][head] = relation
    return matrix

def _restore_ud_syntax(deps_ud: list[list[str]]) -> tuple[list[int], list[str]]:
    heads = [-1] * len(deps_ud)
    deprels = [NO_ARC_LABEL] * len(deps_ud)
    for edge_to, word_rels in enumerate(deps_ud):
        edges_deprels = [
            (edge_from, deprel)
            for edge_from, deprel in enumerate(word_rels) if deprel != NO_ARC_LABEL
        ]
        assert len(edges_deprels) <= 1, f"Basic syntax must have no more than one dependent"
        # "All deprels are empty" indicates current node is null - skip it.
        if len(edges_deprels) == 0:
            continue
        edge_from, deprel = edges_deprels[0]
        heads[edge_to] = edge_from if edge_from != edge_to else 0
        deprels[edge_to] = deprel
    return heads, deprels

# Enhanced syntax.

def _deps_str_to_dict(deps_str: str) -> dict[str, str]:
    """
    Example:
    >>> _deps_str_to_dict("26:conj|18:advcl:while")
    {'26': 'conj', '18': 'advcl:while'}
    """
    if deps_str == '_':
        return dict()

    deps = {}
    for dep in deps_str.split('|'):
        head, rel = dep.split(':', 1)
        assert head not in deps, "Multiedges are not allowed"
        deps[head] = rel
    return deps

def _deps_dict_to_str(deps: dict[str, str]) -> str:
    """
    Inverse function to _dep_str_to_dict.
    """
    if len(deps) == 0:
        return '_'
    return '|'.join([f"{head}:{rel}" for head, rel in deps.items()])

def _renumerate_deps(old2new_id: dict[str, int], deps: list[dict]) -> list[dict]:
    return [{old2new_id[head]: rels for head, rels in dep.items()} for dep in deps]

def _build_syntax_matrix_eud(deps: list[dict[int, str]]) -> list[list[str]]:
    sequence_length = len(deps)

    matrix = [[NO_ARC_LABEL] * sequence_length for _ in range(sequence_length)]
    for index, dep in enumerate(deps):
        assert 0 < len(dep), f"Deps must not be empty"
        for head, relation in dep.items():
            assert 0 <= head
            # Same trick as in basic syntax case.
            if head == 0:
                head = index
            else:
                head -= 1
                assert head != index, f"head = {head + 1} must not be equal to index = {index + 1}"
            matrix[index][head] = relation
    return matrix

def _restore_eud_syntax(deps_eud: list[list[str]]) -> list[dict[str, str]]:
    deps = []
    for edge_to, word_rels in enumerate(deps_eud):
        token_deps = {
            (edge_from if edge_from != edge_to else 0): deprel
            for edge_from, deprel in enumerate(word_rels) if deprel != ''
        }
        deps.append(token_deps)
    return deps


def preprocess_labels(
    ids: list[str],
    words: list[str],
    lemmas: list[str] | None,
    upos: list[str] | None,
    xpos: list[str] | None,
    feats: list[str] | None,
    heads: list[str] | None,
    deprels: list[list[str]] | None,
    deps: list[list[str]] | None,
    miscs: list[str] | None,
    deepslots: list[str] | None,
    semclasses: list[str] | None
):
    result = {"words": words}

    if lemmas is not None:
        result["lemma_rules"] = _build_lemma_rules(words, lemmas)

    if upos is not None and xpos is not None and feats is not None:
        result["joint_pos_feats"] = _join_pos_and_feats(upos, xpos, feats)

    # Map old ids to new ones, so that #NULLs get integer ids
    # (e.g. [1, 1.1, 2] turns into [1, 2, 3]).
    # -1 accounts for `_`, while 0 accounts for `ROOT`.
    old2new_id = {'-1': -1, '0': 0} | {old_id: new_id for new_id, old_id in enumerate(ids, 1)}

    if heads is not None and deprels is not None:
        heads: list[dict[int, str]] = _renumerate_heads(old2new_id, heads)
        result["deps_ud"] = _build_syntax_matrix_ud(heads, deprels)

    if deps is not None:
        deps: list[dict[str, str]] = [_deps_str_to_dict(dep) for dep in deps]
        deps: list[dict[int, str]] = _renumerate_deps(old2new_id, deps)
        result["deps_eud"] = _build_syntax_matrix_eud(deps)

    if miscs is not None:
        result["miscs"] = miscs

    if deepslots is not None:
        result["deepslots"] = deepslots

    if semclasses is not None:
        result["semclasses"] = semclasses

    return result


def postprocess_labels(
    words: list[str],
    lemma_rules: list[str],
    joint_pos_feats: list[str],
    deps_ud: list[list[str]],
    deps_eud: list[list[str]],
    miscs: list[str],
    deepslots: list[str],
    semclasses: list[str]
) -> dict[str, list]:
    ids = _restore_ids(words)
    lemmas = _restore_lemmas(words, lemma_rules)
    upos, xpos, feats = _split_pos_and_feats(joint_pos_feats)

    # Renumerate heads back, e.g. [1, 2, 3] into [1, 1.1, 2].
    # Luckily, we already have this mapping stored in `ids`.
    new2old_id = {-1: '-1', 0: '0'} | {i: id for i, id in enumerate(ids, 1)}
    # Basic syntax.
    heads, deprels = _restore_ud_syntax(deps_ud)
    heads = _renumerate_heads(new2old_id, heads)
    # Enhanced syntax.
    deps = _restore_eud_syntax(deps_eud)
    deps = _renumerate_deps(new2old_id, deps)
    deps = [_deps_dict_to_str(token_deps) for token_deps in deps]

    # Manually set specific tags for nulls.
    for i in range(len(words)):
        if words[i] == "#NULL":
            heads[i] = '_'
            deprels[i] = '_'
            miscs[i] = 'ellipsis'

    return {
        "ids": ids,
        "forms": words,
        "lemmas": lemmas,
        "upos": upos,
        "xpos": xpos,
        "feats": feats,
        "heads": heads,
        "deprels": deprels,
        "deps": deps,
        "miscs": miscs,
        "deepslots": deepslots,
        "semclasses": semclasses
    }
