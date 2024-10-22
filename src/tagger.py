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
from .lemma_classifier import LemmaClassifier
from .lemmatize_helper import LemmaRule, predict_lemma_from_rule
from .dependency_classifier import DependencyClassifier
from .utils import get_null_mask


class Tagger(Model):
    """
    Joint morpho-syntactic-semantic tagger.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        indexer: TokenIndexer,
        embedder: TokenEmbedder,
        lemma_rule_classifier_args: dict,
        pos_feats_classifier_args: dict,
        depencency_classifier_args: dict,
        misc_classifier_args: dict,
        semslot_classifier_args: dict,
        semclass_classifier_args: dict
    ):
        super().__init__()
        self.vocab = vocab

        self.embedder = embedder
        embedding_dim = self.embedder.get_output_dim()

        self.lemma_rule_classifier = LemmaClassifier(
            **lemma_rule_classifier_args,
            in_dim=embedding_dim,
            n_classes=vocab.get_vocab_size("lemma_rule_labels"),
        )
        self.pos_feats_classifier = FeedForwardClassifier(
            **pos_feats_classifier_args,
            in_dim=embedding_dim,
            n_classes=vocab.get_vocab_size("pos_feats_labels"),
        )
        self.dependency_classifier = DependencyClassifier(
            **depencency_classifier_args,
            in_dim=embedding_dim,
            n_rel_classes_ud=vocab.get_vocab_size("deprel_labels"),
            n_rel_classes_eud=vocab.get_vocab_size("deps_labels"),
        )
        self.misc_classifier = FeedForwardClassifier(
            **misc_classifier_args,
            in_dim=embedding_dim,
            n_classes=vocab.get_vocab_size("misc_labels"),
        )
        self.semslot_classifier = FeedForwardClassifier(
            **semslot_classifier_args,
            in_dim=embedding_dim,
            n_classes=vocab.get_vocab_size("semslot_labels"),
        )
        self.semclass_classifier = FeedForwardClassifier(
            **semclass_classifier_args,
            in_dim=embedding_dim,
            n_classes=vocab.get_vocab_size("semclass_labels"),
        )

    @override(check_signature=False)
    def forward(
        self,
        words: TextFieldTensors,
        sentences: List[List[Token]],
        null_mask: BoolTensor, # null_mask[i] = 0 iff i-th token is null.
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

        device = get_device_of(words["tokens"]["token_ids"])

        # [batch_size, seq_len, embedding_dim]
        embeddings = self.embedder(**words['tokens'])
        # Padding mask.
        mask = get_text_field_mask(words)

        # Mask nulls, since they have trivial lemmas.
        lemma_rule = self.lemma_rule_classifier(embeddings, lemma_rule_labels, mask & (~null_mask), metadata)
        # Don't mask nulls, as they actually have non-trivial grammatical features we want to learn.
        pos_feats = self.pos_feats_classifier(embeddings, pos_feats_labels, mask)
        # Mask nulls for basic UD and don't mask for E-UD.
        syntax = self.dependency_classifier(embeddings, deprel_labels, deps_labels, mask & (~null_mask), mask, sentences)
        misc = self.misc_classifier(embeddings, misc_labels, mask)
        semslot = self.semslot_classifier(embeddings, semslot_labels, mask)
        semclass = self.semclass_classifier(embeddings, semclass_labels, mask)

        self._maybe_log_preds_and_probs("Semantic slots:", semslot['probs'][0], "semslot_labels", sentences)
        self._maybe_log_preds_and_probs("Semantic classes:", semclass['probs'][0], "semclass_labels", sentences)

        loss = lemma_rule['loss'] \
            + pos_feats['loss'] \
            + syntax['arc_loss_ud'] \
            + syntax['rel_loss_ud'] \
            + syntax['arc_loss_eud'] \
            + syntax['rel_loss_eud'] \
            + misc['loss'] \
            + semslot['loss'] \
            + semclass['loss']

        return {
            'lemma_rule_preds': lemma_rule['preds'],
            'pos_feats_preds': pos_feats['preds'],
            'syntax_ud': syntax['syntax_ud'],
            'syntax_eud': syntax['syntax_eud'],
            'misc_preds': misc['preds'],
            'semslot_preds': semslot['preds'],
            'semclass_preds': semclass['preds'],
            'loss': loss,
        }

    @override
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # Morphology.
        lemma_accuracy = self.lemma_rule_classifier.get_metrics(reset)['Accuracy']
        pos_feats_accuracy = self.pos_feats_classifier.get_metrics(reset)['Accuracy']
        # Syntax.
        syntax_metrics = self.dependency_classifier.get_metrics(reset)
        uas_ud = syntax_metrics['UD-UAS']
        las_ud = syntax_metrics['UD-LAS']
        uas_eud = syntax_metrics['EUD-UAS']
        las_eud = syntax_metrics['EUD-LAS']
        # Misc
        misc_accuracy = self.misc_classifier.get_metrics(reset)['Accuracy']
        # Semantic.
        semslot_accuracy = self.semslot_classifier.get_metrics(reset)['Accuracy']
        semclass_accuracy = self.semclass_classifier.get_metrics(reset)['Accuracy']
        # Average.
        mean_accuracy = np.mean([
            lemma_accuracy,
            pos_feats_accuracy,
            uas_ud,
            las_ud,
            uas_eud,
            las_eud,
            misc_accuracy,
            semslot_accuracy,
            semclass_accuracy
        ])
        return {
            'Lemma': lemma_accuracy,
            'PosFeats': pos_feats_accuracy,
            'UD-UAS': uas_ud,
            'UD-LAS': las_ud,
            'EUD-UAS': uas_eud,
            'EUD-LAS': las_eud,
            'Misc': misc_accuracy,
            'SS': semslot_accuracy,
            'SC': semclass_accuracy,
            'Avg': mean_accuracy
        }

    @override(check_signature=False)
    def make_output_human_readable(self, output: Dict[str, Tensor]) -> Dict[str, list]:
        sentences = output["sentences"]
        # Make sure batch_size is 1 during prediction
        assert len(sentences) == 1
        sentence = sentences[0]

        # Restore ids.
        ids = [token.id for token in sentence]
        # Restore forms.
        words = [token.form for token in sentence]

        # Decode lemmas.
        lemmas = []
        lemma_rule_strings = self._decode_predictions(output["lemma_rule_preds"][0], "lemma_rule_labels")
        for word, lemma_rule_str in zip(words, lemma_rule_strings):
            if lemma_rule_str == DEFAULT_OOV_TOKEN:
                lemma = word
            else:
                lemma_rule = LemmaRule.from_str(lemma_rule_str)
                lemma = predict_lemma_from_rule(word, lemma_rule)
            lemmas.append(lemma)

        # Decode pos and feats tags.
        upos_tags = []
        xpos_tags = []
        feats_tags = []
        pos_feats_preds = self._decode_predictions(output["pos_feats_preds"][0], "pos_feats_labels")
        for pos_feats_glued in pos_feats_preds:
            if pos_feats_glued == '_':
                upos_tag, xpos_tag, feats_tag = '_', '_', '_'
            else:
                upos_tag, xpos_tag, feats_tag = pos_feats_glued.split('#')
            upos_tags.append(upos_tag)
            xpos_tags.append(xpos_tag)
            feats_tags.append(feats_tag)

        # Decode heads and deprels.
        # Recall that syntactic nodes are renumerated so that [1, 1.1, 2] becomes [1, 2, 3].
        # Now we want to bind them back to the actual tokens, i.e. renumerate [1, 2, 3] into [1, 1.1, 2].
        # Luckily, we already have this mapping stored in 'ids'.
        heads = [None for _ in range(len(sentence))]
        deprels = [None for _ in range(len(sentence))]
        for batch_index, edge_to, edge_from, deprel_id in output["syntax_ud"].tolist():
            # Make sure UD-heads are unique (have no collisions).
            assert heads[edge_to] is None
            assert deprels[edge_to] is None
            # Renumerate nodes back to actual tokens' ids and replace self-loops with ROOT (0).
            heads[edge_to] = ids[edge_from] if edge_from != edge_to else 0
            deprels[edge_to] = self.vocab.get_token_from_index(deprel_id, "deprel_labels")

        # Decode deps.
        deps = [[] for _ in range(len(sentence))]
        for batch_index, edge_to, edge_from, dep_id in output["syntax_eud"].tolist():
            dep = self.vocab.get_token_from_index(dep_id, "deps_labels")
            # Renumerate nodes back to actual tokens' ids and replace self-loops with ROOT (0).
            deps[edge_to].append(f"{ids[edge_from] if edge_from != edge_to else 0}:{dep}")
        deps = ['|'.join(dep) if dep else '_' for dep in deps]

        miscs = self._decode_predictions(output["misc_preds"][0], "misc_labels")
        semslots = self._decode_predictions(output["semslot_preds"][0], "semslot_labels")
        semclasses = self._decode_predictions(output["semclass_preds"][0], "semclass_labels")

        # Manually post-process nulls' tags.
        for i, token in enumerate(sentence):
            if token.is_null():
                heads[i] = '_'
                deprels[i] = '_'
                miscs[i] = 'ellipsis'

        return {
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

