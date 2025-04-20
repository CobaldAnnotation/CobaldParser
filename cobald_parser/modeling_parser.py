from torch import LongTensor
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

from .configuration import CobaldParserConfig
from .encoder import MaskedLanguageModelEncoder
from .mlp_classifier import MlpClassifier
from .dependency_classifier import DependencyClassifier
from .utils import (
    build_padding_mask,
    build_null_mask,
    prepend_cls,
    remove_nulls,
    add_nulls
)


@dataclass
class CobaldParserOutput(ModelOutput):
    """
    Output type for CobaldParser.
    """
    loss: float = None
    words: list = None
    counting_mask: LongTensor = None
    lemma_rules: LongTensor = None
    morph_feats: LongTensor = None
    deps_ud: LongTensor = None
    deps_eud: LongTensor = None
    miscs: LongTensor = None
    deepslots: LongTensor = None
    semclasses: LongTensor = None


class CobaldParser(PreTrainedModel):
    """Morpho-Syntax-Semantic Parser."""

    config_class = CobaldParserConfig

    def __init__(self, config: CobaldParserConfig):
        super().__init__(config)

        self.encoder = MaskedLanguageModelEncoder(
            model_name=config.encoder_model_name
        )
        embedding_size = self.encoder.get_embedding_size()

        self.null_classifier = MlpClassifier(
            input_size=self.encoder.get_embedding_size(),
            hidden_size=config.null_classifier_hidden_size,
            n_classes=config.consecutive_null_limit + 1,
            activation=config.activation,
            dropout=config.dropout
        )
        self.lemma_rule_classifier = MlpClassifier(
            input_size=embedding_size,
            hidden_size=config.lemma_classifier_hidden_size,
            n_classes=len(config.id2lemma_rule),
            activation=config.activation,
            dropout=config.dropout
        )
        self.morph_feats_classifier = MlpClassifier(
            input_size=embedding_size,
            hidden_size=config.morphology_classifier_hidden_size,
            n_classes=len(config.id2morph_feats),
            activation=config.activation,
            dropout=config.dropout
        )
        self.dependency_classifier = DependencyClassifier(
            input_size=embedding_size,
            hidden_size=config.dependency_classifier_hidden_size,
            n_rels_ud=len(config.id2rel_ud),
            n_rels_eud=len(config.id2rel_eud),
            activation=config.activation,
            dropout=config.dropout
        )
        self.misc_classifier = MlpClassifier(
            input_size=embedding_size,
            hidden_size=config.misc_classifier_hidden_size,
            n_classes=len(config.id2misc),
            activation=config.activation,
            dropout=config.dropout
        )
        self.deepslot_classifier = MlpClassifier(
            input_size=embedding_size,
            hidden_size=config.deepslot_classifier_hidden_size,
            n_classes=len(config.id2deepslot),
            activation=config.activation,
            dropout=config.dropout
        )
        self.semclass_classifier = MlpClassifier(
            input_size=embedding_size,
            hidden_size=config.semclass_classifier_hidden_size,
            n_classes=len(config.id2semclass),
            activation=config.activation,
            dropout=config.dropout
        )

    def forward(
        self,
        words: list[list[str]],
        counting_mask: LongTensor = None,
        lemma_rules: LongTensor = None,
        morph_feats: LongTensor = None,
        deps_ud: LongTensor = None,
        deps_eud: LongTensor = None,
        miscs: LongTensor = None,
        deepslots: LongTensor = None,
        semclasses: LongTensor = None,
        sent_ids: list[str] = None,
        texts: list[str] = None,
        inference_mode: bool = False
    ) -> CobaldParserOutput:
        
        # Extra [CLS] token accounts for the case when #NULL is the first token in a sentence.
        words_with_cls = prepend_cls(words)
        words_without_nulls = remove_nulls(words_with_cls)
        # Embeddings of words without nulls.
        embeddings_without_nulls = self.encoder(words_without_nulls)
        # Predict nulls.
        null_output = self.null_classifier(embeddings_without_nulls, counting_mask)

        # "Teacher forcing": during training, pass the original words (with gold nulls)
        # to the classification heads, so that they are trained upon correct sentences.
        if inference_mode:
            # Restore predicted nulls in the original sentences.
            words_with_nulls = add_nulls(words, null_output["preds"])
        else:
            words_with_nulls = words

        # Encode words with nulls.
        # [batch_size, seq_len, embedding_size]
        embeddings = self.encoder(words_with_nulls)

        # Predict lemmas and morphological features.
        lemma_output = self.lemma_rule_classifier(embeddings, lemma_rules)
        morph_feats_output = self.morph_feats_classifier(embeddings, morph_feats)

        # Predict syntax.
        # [batch_size, seq_len]
        padding_mask = build_padding_mask(words_with_nulls, self.device)
        null_mask = build_null_mask(words_with_nulls, self.device)
        # Mask nulls for basic UD and don't mask for E-UD.
        deps_output = self.dependency_classifier(
            embeddings,
            deps_ud,
            deps_eud,
            mask_ud=(padding_mask & ~null_mask),
            mask_eud=padding_mask
        )
        misc_output = self.misc_classifier(embeddings, miscs)
        # Predict semantics.
        deepslot_output = self.deepslot_classifier(embeddings, deepslots)
        semclass_output = self.semclass_classifier(embeddings, semclasses)

        # Add up heads losses.
        loss = (
            null_output['loss'] +
            lemma_output['loss'] +
            morph_feats_output['loss'] +
            deps_output['loss_ud'] +
            deps_output['loss_eud'] +
            misc_output['loss'] +
            deepslot_output['loss'] +
            semclass_output['loss']
        )

        return CobaldParserOutput(
            loss=loss,
            words=words_with_nulls,
            counting_mask=null_output['preds'],
            lemma_rules=lemma_output['preds'],
            morph_feats=morph_feats_output['preds'],
            deps_ud=deps_output['preds_ud'],
            deps_eud=deps_output['preds_eud'],
            miscs=misc_output['preds'],
            deepslots=deepslot_output['preds'],
            semclasses=semclass_output['preds']
        )