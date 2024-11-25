from conllu.models import TokenList

from .token import Token


class Sentence:
    def __init__(self, tokens: list[Token], metadata: dict, renumerate: bool = True, prohibit_range_tokens: bool = True):
        self._tokens = tokens
        self._metadata = metadata
        if prohibit_range_tokens:
            for token in tokens:
                assert '-' not in token.id, \
                    "Range tokens (\d-\d) are not allowed in model. Probably you forgot to call preprocessing.py."
        if renumerate:
            self._renumerate_tokens(self._tokens)

    @staticmethod
    def from_conllu(conllu_tokens: TokenList, *args, **kwargs) -> 'Sentence':
        """
        Create Sentence from TokenList.
        """
        tokens = [Token(**conllu_token) for conllu_token in conllu_tokens]
        metadata = conllu_tokens.metadata
        return Sentence(tokens, metadata, *args, **kwargs)

    def __getitem__(self, index: int) -> Token:
        return self._tokens[index]

    def __len__(self) -> int:
        return len(self._tokens)

    @property
    def tokens(self) -> list[Token]:
        return self._tokens

    @property
    def ids(self) -> list[int]:
        return self._collect_nullable_field("ids")

    @property
    def words(self) -> list[str]:
        return [token.form for token in self._tokens]

    @property
    def lemmas(self) -> list[str] | None:
        return self._collect_nullable_field("lemma")

    @property
    def upos(self) -> list[str] | None:
        return self._collect_nullable_field("upos")

    @property
    def xpos(self) -> list[str] | None:
        return self._collect_nullable_field("xpos")

    @property
    def feats(self) -> list[str] | None:
        return self._collect_nullable_field("feats")

    @property
    def heads(self) -> list[int] | None:
        return self._collect_nullable_field("head")

    @property
    def deprels(self) -> list[str] | None:
        return self._collect_nullable_field("deprel")

    @property
    def deps(self) -> list[dict] | None:
        return self._collect_nullable_field("deps")

    @property
    def miscs(self) -> list[str] | None:
        return self._collect_nullable_field("misc")

    @property
    def deepslots(self) -> list[str] | None:
        return self._collect_nullable_field("deepslot")

    @property
    def semclasses(self) -> list[str] | None:
        return self._collect_nullable_field("semclass")

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def sent_id(self) -> str:
        return self._metadata["sent_id"]

    def serialize(self) -> str:
        lines = []

        if self._metadata:
            for key, value in self._metadata.items():
                if value:
                    line = f"# {key} = {value}"
                else:
                    line = f"# {key}"
                lines.append(line)

        lines += [token.serialize() for token in self._tokens]
        return '\n'.join(lines) + "\n\n"

    def _collect_nullable_field(self, field_type: str) -> list | None:
        field_values = [getattr(token, field_type) for token in self._tokens]

        # If all fields are None, return None (=no reference labels).
        if all(field is None for field in field_values):
            return None
        return field_values

    @staticmethod
    def _renumerate_tokens(tokens: list[Token]):
        """
        Renumerate tokens, so that #NULLs get integer id (e.g. [1, 1.1, 2] turns into [1, 2, 3]).
        This also renumerates 'head' and 'deps' tags accordingly.
        Inplace function.
        """
        old2new_id: dict[str, int] = {'-1': -1, '0': 0} # -1 accounts for _ head and 0 accounts for ROOT head.

        # Change ids.
        for i, token in enumerate(tokens, 1):
            old_id = token.id
            new_id = i
            old2new_id[old_id] = new_id
            token.id = str(new_id)

        # Change heads and deps.
        for i, token in enumerate(tokens, 1):
            if token.head is not None:
                token.head = old2new_id[str(token.head)]
            new_deps = {}
            if token.deps is None:
                # Special case when deps is empty.
                continue
            for head, rels in token.deps.items():
                new_deps[old2new_id[str(head)]] = rels
            token.deps = new_deps

