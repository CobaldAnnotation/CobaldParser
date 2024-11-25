import torch
from torch import Tensor

from utils import recursive_find_unique, recursive_replace


class Vocabulary:
    def __init__(self, samples: list, namespaces: list[str]):
        self._namespaces = set(namespaces)
        self._str2idx = dict()
        self._idx2str = dict()
        for namespace in namespaces:
            # Collect unique labels.
            unique_labels = recursive_find_unique([sample[namespace] for sample in samples])
            # Ensure consistent ordering of labels.
            unique_labels = sorted(unique_labels)
            self._str2idx[namespace] = {label: i for i, label in enumerate(unique_labels)}
            self._idx2str[namespace] = {i: label for i, label in enumerate(unique_labels)}

    def encode(self, sample: dict) -> dict[str, Tensor | list]:
        # Do not modify sample in-place! Use new dict instead.
        encoded_sample = dict()

        for namespace, labels in sample.items():
            if namespace in self._namespaces:
                indexes = recursive_replace(
                    labels,
                    lambda label: self._get_index_from_label(label, namespace)
                )
                encoded_sample[namespace] = torch.tensor(indexes)
            else:
                encoded_sample[namespace] = labels
        return encoded_sample

    def decode(self, encoded_sample: dict) -> dict[str, list]:
        decoded_sample = dict()

        for namespace, indexes in encoded_sample.items():
            if namespace in self._namespaces:
                # Avoid iterating over tensors.
                assert isinstance(indexes, Tensor)
                indexes = indexes.tolist()
                decoded_sample[namespace] = recursive_replace(
                    indexes,
                    lambda index: self._get_label_from_index(index, namespace)
                )
            else:
                decoded_sample[namespace] = indexes
        return decoded_sample

    def replace_index(self, label: str, new_index: int, namespace: str):
        # Make sure new index does not collide with another indexes.
        assert new_index not in self._str2idx[namespace]
        # Delete old index.
        old_index = self._str2idx[namespace][label]
        self._idx2str[namespace].pop(old_index)
        # Set new index.
        self._str2idx[namespace][label] = new_index
        self._idx2str[namespace][new_index] = label

    def get_namespace_size(self, namespace: str) -> int:
        return len(self._str2idx[namespace])

    def _get_index_from_label(self, label: str, namespace: str) -> str:
        return self._str2idx[namespace][label]

    def _get_label_from_index(self, index: int, namespace: str) -> str:
        return self._idx2str[namespace][index]


