from typing import List, Dict

import numpy as np

import torch
from torch import nn
from torch import Tensor, LongTensor

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import TextField
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Token as AllenToken
from allennlp.data import TextFieldTensors
from allennlp.nn.util import get_text_field_mask, move_to_device, get_device_of, get_lengths_from_binary_sequence_mask
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure

import sys
sys.path.append("..")
from common.token import Token, CLS_TOKEN

from .feedforward_classifier import FeedForwardClassifier


@Model.register('null_classifier')
class NullClassifier(FeedForwardClassifier):
    """
    Binary classifier of ellipted tokens.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        indexer: TokenIndexer,
        embedder: TokenEmbedder,
        in_dim: int,
        hid_dim: int,
        activation: str,
        dropout: float,
        consecutive_null_limit: int,
        class_weights: List[float] = None
    ):
        super().__init__(
            vocab=vocab,
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_classes=consecutive_null_limit + 1,
            activation=activation,
            dropout=dropout
        )
        self.indexer = indexer
        self.embedder = embedder

        class_weights = Tensor(class_weights) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = CategoricalAccuracy()
        self.fscore = FBetaMeasure()

    def forward(
        self,
        words: TextFieldTensors,
        sentences: List[List[Token]],
        is_inference: bool
    ) -> Dict[str, Tensor]:

        device = get_device_of(words["tokens"]["token_ids"])

        words_with_nulls = words
        sentences_with_nulls = sentences
        # Extra [CLS] token accounts for the case when #NULL is the first token in a sentence.
        sentences_with_cls_and_nulls = self._add_cls_token(sentences_with_nulls)
        sentences_with_cls_and_no_nulls = self._remove_nulls(sentences_with_cls_and_nulls)

        target_null_counts = self._build_null_count_labels(sentences_with_cls_and_nulls)
        target_null_counts = move_to_device(target_null_counts, device)

        words_with_cls_and_no_nulls = self._create_words(sentences_with_cls_and_no_nulls, device)
        mask_no_nulls = get_text_field_mask(words_with_cls_and_no_nulls)
        embeddings_with_cls_and_no_nulls = self.embedder(**words_with_cls_and_no_nulls['tokens'])
        nulls = super().forward(embeddings_with_cls_and_no_nulls, target_null_counts, mask_no_nulls)
        loss = nulls["loss"]

        if is_inference:
            sentences_with_cls_and_nulls = self._add_nulls(sentences_with_cls_and_no_nulls, nulls["preds"])
            sentences_with_nulls = self._remove_cls_token(sentences_with_cls_and_nulls)
            words_with_nulls = self._create_words(sentences_with_nulls, device=device)

        return words_with_nulls, sentences_with_nulls, loss

    def _create_words(self, sentences: List[List[Token]], device) -> Dict[str, Tensor]:
        text_fields = []
        max_padding_lengths = {}

        for sentence in sentences:
            tokens = [AllenToken(token.form) for token in sentence]
            text_field = TextField(tokens, {"tokens": self.indexer})
            text_field.index(self.vocab)
            if not max_padding_lengths:
                max_padding_lengths = text_field.get_padding_lengths()
            for name, length in text_field.get_padding_lengths().items():
                max_padding_lengths[name] = max(max_padding_lengths[name], length)
            text_fields.append(text_field)

        tensors = []
        for text_field in text_fields:
            tensor = text_field.as_tensor(max_padding_lengths)
            tensors.append(move_to_device(tensor, device))
        assert 0 < len(text_fields)
        words = text_fields[0].batch_tensors(tensors)
        return words

    @staticmethod
    def _build_null_count_labels(sentences: List[List[Token]]) -> LongTensor:
        """
        Count the number of nulls following each token for a bunch of sentences.
        output[i, j] = N mean j-th token in i-th sentence in followed by N nulls

        Example:
        >>> sentences = [
        ...     ['Iraq', 'are', 'reported', 'dead', 'and', '500', '#NULL', '#NULL', '#NULL', 'wounded']
        ... ]
        >>> _build_null_count_labels(sentences)
        [0, 0, 0, 0, 0, 0, 3, 0]
        """
        sentences_null_counts: List[LongTensor] = []
        for sentence in sentences:
            sentence_null_counts = []
            cumulative_null_counter = 0
            for token in sentence:
                if token.is_null():
                    cumulative_null_counter += 1
                else:
                    if 1 <= cumulative_null_counter:
                        assert 1 <= len(sentence_null_counts) and sentence_null_counts[-1] == 0
                        sentence_null_counts[-1] = cumulative_null_counter
                        cumulative_null_counter = 0
                    sentence_null_counts.append(0)
            sentences_null_counts.append(LongTensor(sentence_null_counts))

        null_count_labels = torch.nn.utils.rnn.pad_sequence(sentences_null_counts, batch_first=True, padding_value=-1)
        return null_count_labels.long()

    @staticmethod
    def _remove_nulls(sentences: List[List[Token]]):
        return [[token for token in sentence if not token.is_null()] for sentence in sentences]

    @staticmethod
    def _add_cls_token(sentences: List[List[Token]]):
        # Place token on the first position
        return [[CLS_TOKEN, *sentence] for sentence in sentences]

    @staticmethod
    def _remove_cls_token(sentences: List[List[Token]]):
        # Remove first token.
        return [sentence[1:] for sentence in sentences]

    @staticmethod
    def _add_nulls(sentences: List[List[Token]], sentences_null_counts: LongTensor):
        sentences_with_nulls = []
        for sentence, sentence_null_counts in zip(sentences, sentences_null_counts):
            sentence_with_nulls = []
            for token, n_nulls_to_insert in zip(sentence, sentence_null_counts):
                sentence_with_nulls.append(token)
                for i in range(1, n_nulls_to_insert + 1):
                    sentence_with_nulls.append(Token.create_null(id=f"{token.id}.{i}"))
            sentences_with_nulls.append(sentence_with_nulls)
        return sentences_with_nulls

    def update_metrics(self, logits: Tensor, labels: Tensor, mask: Tensor):
        self.accuracy(logits, labels, mask)
        self.fscore(logits, labels, mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
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

