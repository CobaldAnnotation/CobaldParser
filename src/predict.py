from tqdm import tqdm

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader

from src.utils import build_padding_mask


def _unfold_and_mask(batch: dict[str, Tensor | list]) -> list[dict[str, list]]:
    """
    Unfolds dict of lists to list of single-element dicts.
    """
    # Isolate words from the labels.
    words = batch.pop("words")
    # All the keys but `words`, i.e. list like [
    #     (first sample lemma_rules, first sample joint_pos_feats, ..., first sample semclasses),
    #     (second sample lemma_rules, second sample joint_pos_feats, ..., second sample semclasses),
    #     ...
    # ]
    labels_tuples = list(zip(*batch.values()))
    padding_mask = build_padding_mask(words, labels_tuples[0][0].device)

    samples = []
    for sentence_words, labels_tuple, sentence_mask in zip(words, labels_tuples, padding_mask):
        # Mask padding values and convert tensors to lists.
        labels_tuple = [labels[sentence_mask].tolist() for labels in labels_tuple]
        sample = {"words": sentence_words} | dict(zip(batch.keys(), labels_tuple))
        samples.append(sample)
    return samples


def predict(model: nn.Module, dataloader: DataLoader, device) -> dict[str, Tensor | list]:
    model.eval()
    model.to(device)

    predictions = []
    # Disable gradient computation.
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predict"):
            # Move batch to device.
            batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
            output = model(**batch)
            # Remove loss.
            output.pop("loss")
            samples = _unfold_and_mask(output)
            predictions.extend(samples)
    return predictions
