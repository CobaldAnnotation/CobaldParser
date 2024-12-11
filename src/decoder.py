from lemmatize_helper import reconstruct_lemma


def _restore_ids(self, sentence: list[str]) -> list[str]:
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


def decode(
    sentence: list[str],
    lemma_rules: list[str],
    joint_pos_feats: list[str],
    deps_ud: list[list[str]],
    deps_eud: list[list[str]],
    miscs: list[str],
    deepslots: list[str],
    semclasses: list[str]
):
    ids = self._restore_ids(sentence)
    # Decode lemmas.
    lemmas = [reconstruct_lemma(word, lemma_rule_str) for word, lemma_rules in zip(sentence, lemma_rules)]
    # Decode POS and features.
    # Unzip list of tuples (xpos, upos, feats) to a tuple of lists.
    upos, xpos, feats = zip(*[joint_pos_feat.split('#') for joint_pos_feat in joint_pos_feats])

    # Decode heads and deprels.
    heads = []
    deprels = []
    for edge_to, word_rels in enumerate(deps_ud):
        for edge_from, deprel in enumerate(word_rels):
            if deprel == '':
                continue
            # Make sure there is only one UD head for each token.
            assert heads[edge_to] is None
            # Renumerate heads, e.g. [1, 2, 3], into tokens ids, e.g. [1, 1.1, 2].
            # Luckily, we already have this mapping stored in 'ids'.
            heads.append(ids[edge_from] if edge_from != edge_to else 0)
            deprels.append(deprel)

    # Decode deps.
    deps = []
    for edge_to, word_rels in enumerate(deps_eud):
        word_deps = [
            f"{ids[edge_from] if edge_from != edge_to else 0}:{deprel}"
            for edge_from, deprel in enumerate(word_rels) if deprel != ''
        ]
        word_deps_str = '|'.join(word_deps) if word_deps else '_'
        deps.append(word_deps_str)

    # Manually post-process null tags.
    for i in range(len(sentence)):
        if words[i] == "#NULL":
            heads[i] = '_'
            deprels[i] = '_'
            miscs[i] = 'ellipsis'

    return {
        "ids": ids,
        "forms": sentence,
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
