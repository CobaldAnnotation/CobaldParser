import torch
from torch import Tensor
from torch.utils.data import Dataset

import sys
sys.path.append("..")
from common.sentence import Sentence
from common.parse_conllu import parse_conllu_incr

from lemmatize_helper import construct_lemma_rule
from utils import pad_matrices
from dependency_classifier import NO_ARC_VALUE


NO_ARC_LABEL = ""


class CobaldJointDataset(Dataset):

    def __init__(self, conllu_file_path: str, transform = None):
        super().__init__()

        self._samples = []
        with open(conllu_file_path, "r") as file:
            for sentence in parse_conllu_incr(file):
                sample = self._process(sentence)
                self._samples.append(sample)
        self._transform = transform

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = self._samples[index]
        # Apply transformation to a sample if set.
        if self._transform is not None:
            sample = self._transform(sample)
        return sample

    def _process(self, sentence: Sentence) -> dict[str, any]:
        return {
            "words": sentence.words,
            "lemma_rules": self._construct_lemma_rules(sentence.words, sentence.lemmas),
            "joint_pos_feats": self._merge_pos_and_feats(sentence.upos, sentence.xpos, sentence.feats),
            "deps_ud": self._construct_matrix_ud(sentence.heads, sentence.deprels),
            "deps_eud": self._construct_matrix_eud(sentence.deps),
            "miscs": sentence.miscs,
            "deepslots": sentence.deepslots,
            "semclasses": sentence.semclasses,
        }

    @staticmethod
    def _construct_lemma_rules(words: list[str], lemmas: list[str]) -> list[str]:
        if lemmas is None:
            return None

        # TODO: delete if-else??
        return [
            construct_lemma_rule(word if word is not None else '', lemma if lemma is not None else '')
            for word, lemma in zip(words, lemmas, strict=True)
        ]

    @staticmethod
    def _merge_pos_and_feats(upos: list[str], xpos: list[str], feats: list[dict]) -> list[str]:
        if upos is None or xpos is None or feats is None:
            return None

        joint_feats = [
            '|'.join([f"{k}={v}" for k, v in feat.items()]) if 0 < len(feat) else '_'
            for feat in feats
        ]
        joint_pos_feats = [
            f"{token_upos}#{token_xpos}#{token_joint_feats}"
            for token_upos, token_xpos, token_joint_feats in zip(upos, xpos, joint_feats, strict=True)
        ]
        return joint_pos_feats

    @staticmethod
    def _construct_matrix_ud(heads: list[int], deprels: list[str]) -> list[list[str]]:
        if heads is None or deprels is None:
            return None

        sequence_length = len(heads)

        matrix = [[NO_ARC_LABEL] * sequence_length for _ in range(sequence_length)]
        for index, (head, relation) in enumerate(zip(heads, deprels, strict=True)):
            # Skip nulls.
            if head == -1:
                continue
            assert 0 <= head
            # Hack: start indexing at 0 and replace ROOT with self-loop.
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

    @staticmethod
    def _construct_matrix_eud(deps: list[dict[str, str]]) -> list[list[str]]:
        if deps is None:
            return None

        sequence_length = len(deps)

        matrix = [[NO_ARC_LABEL] * sequence_length for _ in range(sequence_length)]
        for index, dep in enumerate(deps):
            assert 0 < len(dep), f"Deps must not be empty"
            for head, relation in dep.items():
                assert 0 <= head
                # Hack: start indexing at 0 and replace ROOT with self-loop.
                # It makes parser implementation much easier.
                if head == 0:
                    # Replace ROOT with self-loop.
                    head = index
                else:
                    # If not ROOT, shift token left.
                    head -= 1
                    assert head != index, f"head = {head + 1} must not be equal to index = {index + 1}"
                matrix[index][head] = relation

        return matrix

    @staticmethod
    def collate_fn(samples: list[dict[str, any]], padding_value: int = NO_ARC_VALUE) -> dict[str, any]:
        """Collate function for dataloader."""

        stack_list_column = lambda column: [sample[column] for sample in samples]

        column_is_empty = lambda column: all(sample[column] is None for sample in samples)
        # Sequence padding function.
        maybe_pad_sequence_column = lambda column: torch.nn.utils.rnn.pad_sequence(
            [sample[column] for sample in samples],
            padding_value=padding_value,
            batch_first=True
        ) if not column_is_empty(column) else None

        # Matrix padding function.
        maybe_pad_matrix_column = lambda column: pad_matrices(
            [samples[column] for samples in samples],
            padding_value=padding_value
        ) if not column_is_empty(column) else None

        return {
            "words": stack_list_column('words'),
            "lemma_rules": maybe_pad_sequence_column('lemma_rules'),
            "joint_pos_feats": maybe_pad_sequence_column('joint_pos_feats'),
            "deps_ud": maybe_pad_matrix_column('deps_ud'),
            "deps_eud": maybe_pad_matrix_column('deps_eud'),
            "miscs": maybe_pad_sequence_column('miscs'),
            "deepslots": maybe_pad_sequence_column('deepslots'),
            "semclasses": maybe_pad_sequence_column('semclasses')
        }

