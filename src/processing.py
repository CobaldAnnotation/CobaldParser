import conllu
from collections import defaultdict

from src.lemmatize_helper import construct_lemma_rule, reconstruct_lemma


NO_ARC_LABEL = ""


def _deps_str_to_dict(deps_str: str) -> dict[str, str]:
    """
    Convert deps from string to dict.

    Example:
    >>> _deps_str_to_dict("26:conj|18:advcl:while")
    {'26': 'conj', '18': 'advcl:while'}
    """
    if deps_str == '_':
        return dict()

    deps = {}
    for dep in deps_str.split('|'):
        head, rel = dep.split(':', 1)
        assert head not in deps, "Multiedges are not allowed in CoBaLD"
        deps[head] = rel
    return deps

def _build_arcs(token_deps: dict[int, str], current_index: int, sentence_length: int) -> list[str]:
    # Array of arcs connecting current token with the others.
    arcs = [NO_ARC_LABEL] * sentence_length
    # Set relations at heads position where heads are present.
    for head, rel in token_deps.items():
        assert head != current_index + 1, f"Self-loops are not allowed in UD stantard"
        if head == -1:
            continue
        # Trick: start indexing at 0 and replace `root` with self-loop.
        # This trick makes it easier to work with matrices.
        arcs[head - 1 if head != 0 else current_index] = rel
    return arcs


def preprocess(sentence: conllu.models.TokenList) -> dict[str, list]:
    ids = [token["id"] for token in sentence]
    # Renumerate ids so that #NULLs get integer ids, e.g. [1, 1.1, 2] turns into [1, 2, 3]).
    # -1 accounts for empty head, while 0 accounts for `root`.
    old2new_id = {'_': -1, '0': 0} | {old_id: new_id for new_id, old_id in enumerate(ids, 1)}

    result = defaultdict(list)
    for index, token in enumerate(sentence):
        result["words"].append(token["word"])

        if "lemma" in token:
            lemma_rule = construct_lemma_rule(token["word"], token["lemma"])
            result["lemma_rules"].append(lemma_rule)

        if "upos" in token and "xpos" in token and "feats" in token:
            joint_pos_feats = '#'.join((token["upos"], token["xpos"], token["feats"]))
            result["joint_pos_feats"].append(joint_pos_feats)

        if "head" in token and "deprel" in token:
            # Renumerate basic syntax head.
            head: int = old2new_id[token["head"]]
            deprel = token["deprel"]
            arcs = _build_arcs({head: deprel}, index, len(sentence))
            result["deps_ud"].append(arcs)

        if "deps" in token:
            deps: dict[str, str] = _deps_str_to_dict(token["deps"])
            # Renumerate enhanced syntax heads.
            deps = {old2new_id[head]: rel for head, rel in deps.items()}
            multiarcs = _build_arcs(deps, index, len(sentence))
            result["deps_eud"].append(multiarcs)

        if "misc" in token:
            result["miscs"].append(token["misc"])

        if "deepslot" in token:
            result["deepslots"].append(token["deepslot"])

        if "semclass" in token:
            result["semclasses"].append(token["semclass"])

    return result


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


def postprocess(
    words: list[str],
    lemma_rules: list[str],
    joint_pos_feats: list[str],
    deps_ud: list[list[str]],
    deps_eud: list[list[str]],
    miscs: list[str],
    deepslots: list[str],
    semclasses: list[str]
) -> list[conllu.models.TokenList]:

    ids = _restore_ids(words)
    # Renumerate heads back, e.g. [1, 2, 3] into [1, 1.1, 2].
    # Luckily, we already have this mapping stored in `ids`.
    new2old_id = {i: id for i, id in enumerate(ids)}

    sentence = conllu.models.TokenList()
    tokens_labels = zip(ids, words, lemma_rules, joint_pos_feats, deps_ud, deps_eud, miscs, deepslots, semclasses)
    for i, token_labels in enumerate(tokens_labels):
        id, word, lemma_rule, joint_pos_feat, word_deps_ud, word_deps_eud, misc, deepslot, semclass = token_labels
        token = {"id": id, "word": word}
        token["lemma"] = reconstruct_lemma(word, lemma_rule)
        token["upos"], token["xpos"], token["feats"] = joint_pos_feat.split('#')

        # Syntax.
        edge_to = i # alias
        collect_heads_and_deps = lambda word_relations: {
            (new2old_id[edge_from] if edge_from != edge_to else 0): deprel
            for edge_from, deprel in enumerate(word_relations) if deprel != NO_ARC_LABEL
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
        sentence.append(token)

    return sentence
