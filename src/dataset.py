import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.read_conllu import read_conllu
from src.processing import preprocess_labels
from src.utils import collect_values, pad_sequences, pad_matrices


class CobaldJointDataset(Dataset):

    def __init__(self, conllu_path: str, transform = None):
        super().__init__()

        self._samples = []
        for sentence in read_conllu(conllu_path):
            columns = sentence[0].keys()
            sample = preprocess_labels(
                collect_values(sentence, "id"),
                collect_values(sentence, "form"),
                collect_values(sentence, "lemma") if "lemma" in columns else None,
                collect_values(sentence, "upos") if "upos" in columns else None,
                collect_values(sentence, "xpos") if "xpos" in columns else None,
                collect_values(sentence, "feats") if "feats" in columns else None,
                collect_values(sentence, "head") if "head" in columns else None,
                collect_values(sentence, "deprel") if "deprel" in columns else None,
                collect_values(sentence, "deps") if "deps" in columns else None,
                collect_values(sentence, "misc") if "misc" in columns else None,
                collect_values(sentence, "deepslot") if "deepslot" in columns else None,
                collect_values(sentence, "semclass") if "semclass" in columns else None
            )
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

    @staticmethod
    def collate_fn(samples: list[dict[str, any]], padding_value: int = -1) -> dict[str, any]:
        """Collate function for dataloader."""

        # Define shortcut functions.
        has_column = lambda column_name: all(column_name in sample for sample in samples)

        result = {"words": collect_values(samples, "words")}

        if has_column("lemma_rules"):
            result["lemma_rules"] = pad_sequences(collect_values(samples, "lemma_rules"), padding_value)

        if has_column("joint_pos_feats"):
            result["joint_pos_feats"] = pad_sequences(collect_values(samples, "joint_pos_feats"), padding_value)

        if has_column("deps_ud"):
            result["deps_ud"] = pad_matrices(collect_values(samples, "deps_ud"), padding_value)

        if has_column("deps_eud"):
            result["deps_eud"] = pad_matrices(collect_values(samples, "deps_eud"), padding_value)

        if has_column("miscs"):
            result["miscs"] = pad_sequences(collect_values(samples, "miscs"), padding_value)

        if has_column("deepslots"):
            result["deepslots"] = pad_sequences(collect_values(samples, "deepslots"), padding_value)

        if has_column("semclasses"):
            result["semclasses"] = pad_sequences(collect_values(samples, "semclasses"), padding_value)

        return result
