import conllu


def parse_nullable_value(value: str) -> str | None:
    return value if value else None

def parse_head(value: str) -> int | None:
    if not value:
        return None
    if value == '_':
        return '-1'
    return value


# Conllu fields to parse.
FIELDS = [
    "id",
    "form",
    "lemma",
    "upos",
    "xpos",
    "feats",
    "head",
    "deprel",
    "deps",
    "misc",
    "deepslot",
    "semclass"
]

# Override default conllu parsing procedures.
FIELD_PARSERS = {
    "id": lambda line, i: line[i], # Do not split indexes like 1.1
    "lemma": lambda line, i: parse_nullable_value(line[i]),
    "upos": lambda line, i: parse_nullable_value(line[i]), # Do not treat _ as None
    "xpos": lambda line, i: parse_nullable_value(line[i]), # Do not treat _ as None
    "feats": lambda line, i: parse_nullable_value(line[i]), # Do not treat _ as None
    "head": lambda line, i: parse_head(line[i]),
    "deps": lambda line, i: parse_nullable_value(line[i]),
    "misc": lambda line, i: parse_nullable_value(line[i])
}

def read_conllu(conllu_path: str) -> list[conllu.models.TokenList]:
    with open(conllu_path, "r") as file:
        sentences = [
            sentence.filter(id=lambda x: '-' not in x) # Remove range tokens.
            for sentence in conllu.parse_incr(
                file,
                fields=FIELDS,
                field_parsers=FIELD_PARSERS
            )
        ]
    return sentences
