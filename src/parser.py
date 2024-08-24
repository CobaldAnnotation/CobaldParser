from overrides import override
from typing import Dict, List

import logging
logger = logging.getLogger("parser")

import numpy as np

import torch
from torch import Tensor, BoolTensor, LongTensor

from allennlp.nn.util import get_text_field_mask, move_to_device, get_device_of
from allennlp.common import Lazy
from allennlp.data import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.models import Model
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.data.token_indexers import TokenIndexer

import sys
sys.path.append("..")
from common.token import Token

from .feedforward_classifier import FeedForwardClassifier
from .null_classifier import NullClassifier
from .utils import get_null_mask


@Model.register('morpho_syntax_semantic_parser')
class MorphoSyntaxSemanticParser(Model):
    """
    Joint Morpho-Syntax-Semantic Parser.
    See https://guide.allennlp.org/your-first-model for guidance.
    """

    # See https://guide.allennlp.org/using-config-files to find more about Lazy.
    #
    # TODO: move Lazy to from_lazy_objects (as here https://guide.allennlp.org/using-config-files#4)
    def __init__(
        self,
        vocab: Vocabulary,
        indexer: TokenIndexer,
        embedder: TokenEmbedder,
        lemma_rule_classifier,
        pos_feats_classifier,
        depencency_classifier,
        misc_classifier,
        semslot_classifier: Lazy[FeedForwardClassifier],
        semclass_classifier: Lazy[FeedForwardClassifier],
        null_classifier: Lazy[NullClassifier]
    ):
        super().__init__(vocab)

        self.embedder = embedder
        embedding_dim = self.embedder.get_output_dim()

        self.semslot_classifier = semslot_classifier.construct(
            in_dim=embedding_dim,
            n_classes=vocab.get_vocab_size("semslot_labels"),
        )
        self.semclass_classifier = semclass_classifier.construct(
            in_dim=embedding_dim,
            n_classes=vocab.get_vocab_size("semclass_labels"),
        )
        self.null_classifier = null_classifier.construct(
            indexer=indexer,
            embedder=embedder,
            in_dim=embedding_dim,
        )

    @override(check_signature=False)
    def forward(
        self,
        words: TextFieldTensors,
        sentences: List[List[Token]],
        lemma_rule_labels: Tensor = None,
        pos_feats_labels: Tensor = None,
        deprel_labels: Tensor = None,
        deps_labels: Tensor = None,
        misc_labels: Tensor = None,
        semslot_labels: Tensor = None,
        semclass_labels: Tensor = None,
        metadata: List[Dict] = None
    ) -> Dict[str, Tensor]:

        self._maybe_log_sentence(metadata)

        # If all labels are empty, we are at inference.
        is_inference = lemma_rule_labels is None \
            and pos_feats_labels is None \
            and deprel_labels is None \
            and deps_labels is None \
            and misc_labels is None  \
            and semslot_labels is None \
            and semclass_labels is None

        device = get_device_of(words["tokens"]["token_ids"])

        words_with_nulls, sentences_with_nulls, null_loss = self.null_classifier(words, sentences, is_inference)

        # [batch_size, seq_len, embedding_dim]
        embeddings = self.embedder(**words_with_nulls['tokens'])
        # Padding mask.
        mask = get_text_field_mask(words_with_nulls)
        # Mask with nulls excluded.
        null_mask = get_null_mask(sentences_with_nulls)
        null_mask = move_to_device(null_mask, device)

        semslot = self.semslot_classifier(embeddings, semslot_labels, mask)
        semclass = self.semclass_classifier(embeddings, semclass_labels, mask)

        self._maybe_log_preds_and_probs("Semantic slots:", semslot['probs'][0], "semslot_labels", sentences_with_nulls)
        self._maybe_log_preds_and_probs("Semantic classes:", semclass['probs'][0], "semclass_labels", sentences_with_nulls)

        loss = semslot['loss'] \
            + semclass['loss'] \
            + null_loss

        return {
            'sentences': sentences_with_nulls,
            'semslot_preds': semslot['preds'],
            'semclass_preds': semclass['preds'],
            'loss': loss,
            'metadata': metadata,
        }

    @override
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # Semantic.
        semslot_accuracy = self.semslot_classifier.get_metrics(reset)['Accuracy']
        semclass_accuracy = self.semclass_classifier.get_metrics(reset)['Accuracy']
        # Average.
        mean_accuracy = np.mean([
            semslot_accuracy,
            semclass_accuracy
        ])
        # Nulls (do not average).
        null_metrics = self.null_classifier.get_metrics(reset)

        return {
            'SS': semslot_accuracy,
            'SC': semclass_accuracy,
            'Avg': mean_accuracy,
            **null_metrics
        }

    @override(check_signature=False)
    def make_output_human_readable(self, output: Dict[str, Tensor]) -> Dict[str, list]:
        sentences = output["sentences"]
        # Make sure batch_size is 1 during prediction
        assert len(sentences) == 1
        sentence = sentences[0]
        sentence_len = len(sentence)

        # Restore ids.
        ids = [token.id for token in sentence]
        # Restore forms.
        words = [token.form for token in sentence]

        # Decode lemmas.
        lemmas = ['_' for _ in range(len(sentence))]

        # Decode pos and feats tags.
        upos_tags = ['_' for _ in range(len(sentence))]
        xpos_tags = ['_' for _ in range(len(sentence))]
        feats_tags = ['_' for _ in range(len(sentence))]

        # Decode heads and deprels.
        heads = ['_' for _ in range(len(sentence))]
        deprels = ['_' for _ in range(sentence_len)]

        # Decode deps.
        deps = [[] for _ in range(len(sentence))]

        miscs = ['_' for _ in range(len(sentence))]
        semslots = self._decode_predictions(output["semslot_preds"][0], "semslot_labels")
        semclasses = self._decode_predictions(output["semclass_preds"][0], "semclass_labels")

        metadata = output["metadata"][0]

        # Manually post-process nulls' tags.
        for i, token in enumerate(sentence):
            if token.is_null():
                heads[i] = '_'
                deprels[i] = '_'
                miscs[i] = 'ellipsis'

        return {
            "ids": [ids],
            "forms": [words],
            "lemmas": [lemmas],
            "upos": [upos_tags],
            "xpos": [xpos_tags],
            "feats": [feats_tags],
            "heads": [heads],
            "deprels": [deprels],
            "deps": [deps],
            "miscs": [miscs],
            "semslots": [semslots],
            "semclasses": [semclasses],
            "metadata": [metadata],
        }

    def _decode_predictions(self, pred_ids: LongTensor, namespace: str):
        """
        Decode classifier predictions (class identifiers) for a sentence using vocabulary.
        E.g. [11, 3, 234, 3] -> ["TO_THINK", "HUMAN", "ANIMAL", "HUMAN"].
        """
        assert len(pred_ids.shape) == 1
        return [self.vocab.get_token_from_index(pred_id, namespace) for pred_id in pred_ids.tolist()]

    @staticmethod
    def _maybe_log_sentence(sentences_metadata: List[Dict]):
        if sentences_metadata is not None and len(sentences_metadata) == 1:
            # Only do log when batch size is 1.
            sentence_metadata = sentences_metadata[0]

            logger.info(f"sent_id = {sentence_metadata['sent_id']}")
            logger.info(f"text = {sentence_metadata['text']}")
            logger.info(f"----------------------------------------------------------")

    def _maybe_log_preds_and_probs(self, message: str, probs: Tensor, namespace: str, sentences: List[Token]):
        """
        Log predictions and their probabilities for each token.
        """
        if len(sentences) == 1:
            # Only do log when batch size is 1.
            sentence = sentences[0]

            # [seq_len, n_classes]
            assert len(probs.shape) == 2
            assert len(probs) == len(sentence)
            # [seq_len]
            pred_probs, pred_ids = torch.max(probs, dim=-1)
            pred_labels = self._decode_predictions(pred_ids, namespace)

            max_token_char_len = max(len(token.form) for token in sentence)
            max_label_char_len = max(len(label) for label in pred_labels)

            logger.info(message)
            logger.info(f"ID\t{'Form':{max_token_char_len}} {'Prediction':{max_label_char_len}} Probability")
            for token, pred_label, pred_prob in zip(sentence, pred_labels, pred_probs):
                logger.info(f"{token.id}\t{token.form:{max_token_char_len}} {pred_label:{max_label_char_len}} {pred_prob:.2f}")
            logger.info(f"----------------------------------------------------------")

