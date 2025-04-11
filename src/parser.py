from torch import LongTensor
from transformers import PretrainedConfig, PreTrainedModel

from src.mlp_classifier import MlpClassifier
from src.dependency_classifier import DependencyClassifier
from src.encoder import MaskedLanguageModelEncoder
from src.utils import (
    build_padding_mask,
    build_null_mask,
    prepend_cls,
    remove_nulls,
    add_nulls
)


class MorphoSyntaxSemanticsParserConfig(PretrainedConfig):
    model_type = "cobald_parser"

    def __init__(
        self,
        encoder_model_name: str = None,
        null_classifier_hidden_size: int = 0,
        lemma_classifier_hidden_size: int = 0,
        morphology_classifier_hidden_size: int = 0,
        dependency_classifier_hidden_size: int = 0,
        misc_classifier_hidden_size: int = 0,
        deepslot_classifier_hidden_size: int = 0,
        semclass_classifier_hidden_size: int = 0,
        consecutive_null_limit: int = 0,
        num_lemmas: int = 0,
        num_morph_feats: int = 0,
        num_rels_ud: int = 0,
        num_rels_eud: int = 0,
        num_miscs: int = 0,
        num_deepslots: int = 0,
        num_semclasses: int = 0,
        activation: str = 'relu',
        dropout: float = 0.1,
        **kwargs
    ):
        self.encoder_model_name = encoder_model_name
        self.null_classifier_hidden_size = null_classifier_hidden_size
        self.consecutive_null_limit = consecutive_null_limit
        self.lemma_classifier_hidden_size = lemma_classifier_hidden_size
        self.morphology_classifier_hidden_size = morphology_classifier_hidden_size
        self.dependency_classifier_hidden_size = dependency_classifier_hidden_size
        self.misc_classifier_hidden_size = misc_classifier_hidden_size
        self.deepslot_classifier_hidden_size = deepslot_classifier_hidden_size
        self.semclass_classifier_hidden_size = semclass_classifier_hidden_size
        self.num_lemmas = num_lemmas
        self.num_morph_feats = num_morph_feats
        self.num_rels_ud = num_rels_ud
        self.num_rels_eud = num_rels_eud
        self.num_miscs = num_miscs
        self.num_deepslots = num_deepslots
        self.num_semclasses = num_semclasses
        self.activation = activation
        self.dropout = dropout
        super().__init__(**kwargs)


class MorphoSyntaxSemanticsParser(PreTrainedModel):
    """Morpho-Syntax-Semantic Parser."""

    def __init__(self, config: MorphoSyntaxSemanticsParserConfig):
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
            n_classes=config.num_lemmas,
            activation=config.activation,
            dropout=config.dropout
        )
        self.morph_feats_classifier = MlpClassifier(
            input_size=embedding_size,
            hidden_size=config.morphology_classifier_hidden_size,
            n_classes=config.num_morph_feats,
            activation=config.activation,
            dropout=config.dropout
        )
        self.dependency_classifier = DependencyClassifier(
            input_size=embedding_size,
            hidden_size=config.dependency_classifier_hidden_size,
            n_rels_ud=config.num_rels_ud,
            n_rels_eud=config.num_rels_eud,
            activation=config.activation,
            dropout=config.dropout
        )
        self.misc_classifier = MlpClassifier(
            input_size=embedding_size,
            hidden_size=config.misc_classifier_hidden_size,
            n_classes=config.num_miscs,
            activation=config.activation,
            dropout=config.dropout
        )
        self.deepslot_classifier = MlpClassifier(
            input_size=embedding_size,
            hidden_size=config.deepslot_classifier_hidden_size,
            n_classes=config.num_deepslots,
            activation=config.activation,
            dropout=config.dropout
        )
        self.semclass_classifier = MlpClassifier(
            input_size=embedding_size,
            hidden_size=config.semclass_classifier_hidden_size,
            n_classes=config.num_semclasses,
            activation=config.activation,
            dropout=config.dropout
        )

    def forward(
        self,
        words: list[list[str]],
        counting_mask: LongTensor = None,
        lemma_rules: LongTensor = None,
        morph_feats: LongTensor = None,
        syntax_ud: LongTensor = None,
        syntax_eud: LongTensor = None,
        miscs: LongTensor = None,
        deepslots: LongTensor = None,
        semclasses: LongTensor = None,
        sent_ids: list[str] = None,
        texts: list[str] = None,
        inference_mode: bool = False
    ) -> dict[str, any]:
        
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
            syntax_ud,
            syntax_eud,
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

        return {
            'words': words_with_nulls,
            'counting_mask': null_output['preds'],
            'lemma_rules': lemma_output['preds'],
            'morph_feats': morph_feats_output['preds'],
            'syntax_ud': deps_output['preds_ud'],
            'syntax_eud': deps_output['preds_eud'],
            'miscs': misc_output['preds'],
            'deepslots': deepslot_output['preds'],
            'semclasses': semclass_output['preds'],
            'loss': loss
        }