import textwrap

import torch
from torch import Tensor

from allennlp.data.fields import Field, SequenceField
from allennlp.data.vocabulary import Vocabulary


class MultilabelAdjacencyField(Field[Tensor]):
    def __init__(
        self,
        sequence_length: int,
        indices: list[tuple[int, int]],
        labels: list[str],
        label_namespace: str = "labels",
        padding_value: int = -1,
    ):
        self.indices = indices
        self.labels = labels
        self._num_labels: int = None
        self._labels_ids: list[list[int]] = None
        self._label_namespace = label_namespace
        self._padding_value = padding_value

        assert len(set(indices)) == len(indices), f"Indices must be unique, but found {indices}"
        assert len(indices) == len(labels), f"Labels and indices lengths do not match: {labels}, {indices}"
        assert all(0 <= index[0] < sequence_length and 0 <= index[1] < sequence_length for index in indices), \
            f"Label indices and sequence length are incompatible: {indices} and {sequence_length}"

    def count_vocab_items(self, counter: dict[str, dict[str, int]]):
        if self._labels_ids is None:
            for label in self.labels:
                counter[self._label_namespace][label] += 1

    def index(self, vocab: Vocabulary):
        if self._labels_ids is None:
            self._labels_ids = [vocab.get_token_index(label, self._label_namespace) for label in self.labels]
        if not self._num_labels:
            self._num_labels = vocab.get_vocab_size(self._label_namespace)

    def get_padding_lengths(self) -> dict[str, int]:
        return {"num_tokens": self.sequence_field.sequence_length()}

    def as_tensor(self, padding_lengths: dict[str, int]) -> Tensor:
        seq_len = padding_lengths["num_tokens"]
        # Initialize all with padding value.
        matrix = torch.full((seq_len, seq_len), self._padding_value, dtype=torch.long)
        assert self._labels_ids is not None
        # Assign labels to edges.
        for index, label_id in zip(self.indices, self._labels_ids):
            matrix[index] = label_id
        return matrix

    def __str__(self):
        length = self.sequence_field.sequence_length()
        formatted_labels = "".join(
            "\t\t" + labels + "\n" for labels in textwrap.wrap(repr(self.labels), 100)
        )
        formatted_indices = "".join(
            "\t\t" + index + "\n" for index in textwrap.wrap(repr(self.indices), 100)
        )
        return (
            f"MultilabelAdjacencyField of length {length}\n"
            f"\t\twith indices:\n {formatted_indices}\n"
            f"\t\tand labels:\n {formatted_labels} \t\tin namespace: '{self._label_namespace}'."
        )

    def __len__(self):
        return len(self.sequence_field)

    def human_readable_repr(self):
        ret = {"indices": self.indices}
        if self.labels is not None:
            ret["labels"] = self.labels
        return ret

