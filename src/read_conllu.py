import conllu


# Conllu fields to parse.
FIELDS = [
    "id",
    "word", # Same as "form".
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

def parse_nullable_value(value: str) -> str | None:
    return value if value else None

# Default conllu parsing procedures treat _ as None.
# We don't want this behavior, so explicitly provide the parsing procudures.
FIELD_PARSERS = {
    "id": lambda line, i: line[i], # Do not split indexes like 1.1
    "lemma": lambda line, i: parse_nullable_value(line[i]),
    "upos": lambda line, i: parse_nullable_value(line[i]),
    "xpos": lambda line, i: parse_nullable_value(line[i]),
    "feats": lambda line, i: parse_nullable_value(line[i]),
    "head": lambda line, i: parse_nullable_value(line[i]),
    "deprel": lambda line, i: parse_nullable_value(line[i]),
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
        #for sentence in sentences:
        #    for token in sentence:
        #        if token["word"] == "#NULL":
        #            print(dict(token))

    return sentences
