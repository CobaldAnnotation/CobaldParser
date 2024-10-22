from overrides import override

import numpy as np

import torch
from torch import nn
from torch import Tensor, BoolTensor, LongTensor

from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure

import sys
sys.path.append("..")
from common.token import Token, CLS_TOKEN

from .mlp import MLP


class NullPredictor(nn.Model):
    """
    A pipeline to restore ellipted tokens.
    """
    def __init__(
        self,
        backbone: Backbone,
        hid_dim: int,
        activation: str,
        dropout: float,
        consecutive_null_limit: int,
        class_weights: list[float] = None
    ):
        self.backbone = backbone

        self.null_classifier = MLP(
            in_dim=self.backbone.get_output_dim(),
            hid_dim=hid_dim,
            n_classes=consecutive_null_limit + 1,
            activation=activation,
            dropout=dropout
        )

        class_weights = Tensor(class_weights) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = CategoricalAccuracy()
        self.fscore = FBetaMeasure()

    def forward(
        self,
        sentences: list[list[Token]],
        is_inference: bool
    ) -> dict[str, Tensor]:

        # Extra [CLS] token accounts for the case when #NULL is the first token in a sentence.
        sentences_with_cls = self._add_cls_token(sentences)
        sentences_with_cls_and_no_nulls = self._remove_nulls(sentences_with_cls)

        embeddings_with_cls_and_no_nulls, mask_no_nulls = self.backbone(sentences_with_cls_and_no_nulls)

        target_counting_masks = self._build_counting_mask(sentences_with_cls) if not is_inference else None
        output = self.null_classifier.forward(embeddings_with_cls_and_no_nulls, target_counting_masks, mask_no_nulls)
        loss = output["loss"]

        if is_inference:
            # Add null to the original sentences.
            counting_mask = output["preds"]
            sentences_with_nulls = self._add_nulls(sentences, counting_mask)
        else:
            # No need to restore nulls during training, just take the input sentences
            sentences_with_nulls = sentences

        return sentences_with_nulls, loss

    @override
    def update_metrics(self, logits: Tensor, labels: LongTensor, mask: BoolTensor):
        self.accuracy(logits, labels, mask)
        self.fscore(logits, labels, mask)

    @override
    def get_metrics(self, reset: bool = False) -> dict[str, float]:
        f1_stat = self.fscore.get_metric(reset)
        metrics = {"NullAccuracy": self.accuracy.get_metric(reset)}

        # Track f1score for each class.
        nonzero_f1 = []
        for i, score in enumerate(f1_stat["fscore"]):
            metrics[f"NullF1/Class={i}"] = score
            if 0.0 < score:
                nonzero_f1.append(score)
        metrics["NullF1/Total"] = np.mean(nonzero_f1)

        return metrics

    ### Private ###

    @staticmethod
    def _build_counting_mask(sentences: list[list[Token]]) -> LongTensor:
        """
        Count the number of nulls following each non-null token for a bunch of sentences.
        output[i, j] = N means j-th non-null token in i-th sentence in followed by N nulls.

        Example:
        >>> sentences = [
        ...     ['Iraq', 'are', 'reported', 'dead', 'and', '500', '#NULL', '#NULL', '#NULL', 'wounded']
        ... ]
        >>> _build_counting_mask(sentences)
        [0, 0, 0, 0, 0, 0, 3, 0]
        """
        counting_masks: list[LongTensor] = []

        for sentence in sentences:
            nonnull_tokens_indices = [i for i, token in enumerate(sentence) if not token.is_null()]
            nonnull_tokens_indices.append(len(sentence))
            nonnull_tokens_indices = torch.LongTensor(nonnull_tokens_indices)
            counting_mask = torch.diff(nonnull_tokens_indices) - 1
            counting_masks.append(counting_mask)

        counting_masks_batched = torch.nn.utils.rnn.pad_sequence(counting_masks, batch_first=True, padding_value=-1)
        return counting_masks_batched.long()

    @staticmethod
    def _remove_nulls(sentences: list[list[Token]]) -> list[list[Token]]:
        """
        Return a copy of sentences with nulls removed.
        """
        return [[token for token in sentence if not token.is_null()] for sentence in sentences]

    @staticmethod
    def _add_cls_token(sentences: list[list[Token]]) -> list[list[Token]]:
        """
        Return a copy of sentences with [CLS] tokens prepended.
        """
        return [[CLS_TOKEN, *sentence] for sentence in sentences]

    @staticmethod
    def _add_nulls(sentences: list[list[Token]], counting_masks: LongTensor) -> list[list[Token]]:
        """
        Return a copy of sentences with nulls restored according to .
        """
        sentences_with_nulls = []
        for sentence, counting_mask in zip(sentences, counting_masks):
            sentence_with_nulls = []
            for token, n_nulls_to_insert in zip(sentence, counting_mask):
                sentence_with_nulls.append(token)
                for i in range(1, n_nulls_to_insert + 1):
                    sentence_with_nulls.append(Token.create_null(id=f"{token.id}.{i}"))
            sentences_with_nulls.append(sentence_with_nulls)
        return sentences_with_nulls

