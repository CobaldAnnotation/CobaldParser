import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.read_conllu import read_conllu
from src.processing import preprocess
from src.utils import collect_values, pad_sequences, pad_matrices


class CobaldJointDataset(Dataset):

    def __init__(self, conllu_path: str, transform = None):
        super().__init__()

        self._samples = []
        for sentence in read_conllu(conllu_path):
            sample = preprocess(sentence)
            self._samples.append(sample)
        self._transform = transform

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = self._samples[index]
        # Apply transformation to a sample if given.
        if self._transform is not None:
            sample = self._transform(sample)
        return sample

    @staticmethod
    def collate_fn(samples: list[dict[str, any]], padding_value: int = -1) -> dict[str, any]:
        """Collate function for dataloader."""

        # Define shortcut functions.
        has_field = lambda field: all(field in sample for sample in samples)

        result = {"words": collect_values(samples, "words")}

        if has_field("lemma_rules"):
            result["lemma_rules"] = pad_sequences(collect_values(samples, "lemma_rules"), padding_value)

        if has_field("joint_pos_feats"):
            result["joint_pos_feats"] = pad_sequences(collect_values(samples, "joint_pos_feats"), padding_value)

        if has_field("deps_ud"):
            result["deps_ud"] = pad_matrices(collect_values(samples, "deps_ud"), padding_value)

        if has_field("deps_eud"):
            result["deps_eud"] = pad_matrices(collect_values(samples, "deps_eud"), padding_value)

        if has_field("miscs"):
            result["miscs"] = pad_sequences(collect_values(samples, "miscs"), padding_value)

        if has_field("deepslots"):
            result["deepslots"] = pad_sequences(collect_values(samples, "deepslots"), padding_value)

        if has_field("semclasses"):
            result["semclasses"] = pad_sequences(collect_values(samples, "semclasses"), padding_value)

        return result
