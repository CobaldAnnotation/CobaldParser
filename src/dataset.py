import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.read_conllu import read_conllu
from src.processing import preprocess
from src.utils import pad_sequences, pad_matrices


class CobaldJointDataset(Dataset):
    def __init__(self, conllu_path: str, transform: callable = None):
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

        result = {"words": [sample["words"] for sample in samples]}

        field_to_padding_fn_map = [
            ("lemma_rules", pad_sequences),
            ("joint_pos_feats", pad_sequences),
            ("deps_ud", pad_matrices),
            ("deps_eud", pad_matrices),
            ("miscs", pad_sequences),
            ("deepslots", pad_sequences),
            ("semclasses", pad_sequences)
        ]
        for field, padding_fn in field_to_padding_fn_map:
            # Pad field if it is present in samples.
            if all(field in sample for sample in samples):
                field_values = [sample[field] for sample in samples]
                result[field] = padding_fn(field_values, padding_value)

        return result
