import numpy as np

import torch
from torch import nn
from torch import tensor

from mlp_classifier import MLPClassifier
from encoder import MaskedLanguageModelEncoder
from utils import build_padding_mask


class NullPredictor(nn.Module):
    """A pipeline to restore ellipted tokens."""
    def __init__(
        self,
        encoder: MaskedLanguageModelEncoder,
        hidden_size: int,
        activation: str,
        dropout: float,
        consecutive_null_limit: int,
        class_weights: list[float] = None
    ):
        super().__init__()

        self.encoder = encoder
        self.null_classifier = MLPClassifier(
            input_size=self.encoder.get_embedding_size(),
            hidden_size=hidden_size,
            n_classes=consecutive_null_limit + 1,
            activation=activation,
            dropout=dropout,
            class_weights=class_weights
        )

    def forward(self, words: list[list[str]], is_inference: bool) -> dict[str, tensor]:
        # Extra [CLS] token accounts for the case when #NULL is the first token in a sentence.
        words_with_cls = self._prepend_cls(words)

        # Delete nulls.
        words_without_nulls = self._del_nulls(words_with_cls)
        # Embeddings of words without nulls.
        embeddings_without_nulls = self.encoder(words_without_nulls)

        # Build padding mask.
        padding_mask_without_nulls = build_padding_mask(words_without_nulls, embeddings_without_nulls.device)

        # Build targets (if not at inference).
        if is_inference:
            gold_counting_mask = None
        else:
            gold_counting_mask = self._build_counting_mask(words_with_cls).to(embeddings_without_nulls.device)

        # Predict counting mask.
        classifier_output = self.null_classifier(
            embeddings_without_nulls,
            gold_counting_mask,
            padding_mask_without_nulls
        )

        # Add predicted nulls to the original sentences.
        pred_counting_mask = classifier_output["preds"]
        words_with_nulls = self._add_nulls(words, pred_counting_mask)
        return {
            "words": words_with_nulls,
            "loss": classifier_output["loss"]
        }

    @staticmethod
    def _build_counting_mask(sentences: list[list[str]]) -> tensor:
        """
        Count the number of nulls following each non-null token for a bunch of sentences.
        output[i, j] = N means j-th non-null token in i-th sentence in followed by N nulls.
        """
        counting_masks: list[tensor] = []

        for sentence in sentences:
            nonnull_words_idxs = [i for i, word in enumerate(sentence) if word != "#NULL"]
            nonnull_words_idxs.append(len(sentence))
            nonnull_words_idxs = torch.tensor(nonnull_words_idxs, dtype=torch.int)
            counting_mask = torch.diff(nonnull_words_idxs) - 1
            counting_masks.append(counting_mask)

        counting_masks_batched = torch.nn.utils.rnn.pad_sequence(
            counting_masks,
            batch_first=True,
            # Use -1 to make sure it is never attended to.
            # (if it is, the CUDA will terminate with an error that classes must be positive).
            padding_value=-1
        )
        return counting_masks_batched.long()

    @staticmethod
    def _prepend_cls(sentences: list[list[str]]) -> list[list[str]]:
        """
        Return a copy of sentences with [CLS] token prepended.
        """
        return [["[CLS]", *sentence] for sentence in sentences]

    @staticmethod
    def _del_nulls(sentences: list[list[str]]) -> list[list[str]]:
        """
        Return a copy of sentences with nulls removed.
        """
        return [[word for word in sentence if word != "#NULL"] for sentence in sentences]

    @staticmethod
    def _add_nulls(sentences: list[list[str]], counting_masks: tensor) -> list[list[str]]:
        """
        Return a copy of sentences with nulls restored according to counting masks.
        """
        sentences_with_nulls = []
        for sentence, counting_mask in zip(sentences, counting_masks):
            sentence_with_nulls = []
            for word, n_nulls_to_insert in zip(sentence, counting_mask):
                sentence_with_nulls.append(word)
                for _ in range(1, n_nulls_to_insert + 1):
                    sentence_with_nulls.append("#NULL")
            sentences_with_nulls.append(sentence_with_nulls)
        return sentences_with_nulls

