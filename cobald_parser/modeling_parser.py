from torch import LongTensor
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

from .configuration import CobaldParserConfig
from .encoder import WordTransformerEncoder
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
    joint_feats: LongTensor = None
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

        self.encoder = WordTransformerEncoder(
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
        if "lemma_rule" in config.vocabulary:
            self.lemma_rule_classifier = MlpClassifier(
                input_size=embedding_size,
                hidden_size=config.lemma_classifier_hidden_size,
                n_classes=len(config.vocabulary["lemma_rule"]),
                activation=config.activation,
                dropout=config.dropout
            )
        if "joint_feats" in config.vocabulary:
            self.morphology_classifier = MlpClassifier(
                input_size=embedding_size,
                hidden_size=config.morphology_classifier_hidden_size,
                n_classes=len(config.vocabulary["joint_feats"]),
                activation=config.activation,
                dropout=config.dropout
            )
        if "ud_deprel" in config.vocabulary or "eud_deprel" in config.vocabulary:
            self.dependency_classifier = DependencyClassifier(
                input_size=embedding_size,
                hidden_size=config.dependency_classifier_hidden_size,
                n_rels_ud=len(config.vocabulary["ud_deprel"]),
                n_rels_eud=len(config.vocabulary["eud_deprel"]),
                activation=config.activation,
                dropout=config.dropout
            )
        if "misc" in config.vocabulary:
            self.misc_classifier = MlpClassifier(
                input_size=embedding_size,
                hidden_size=config.misc_classifier_hidden_size,
                n_classes=len(config.vocabulary["misc"]),
                activation=config.activation,
                dropout=config.dropout
            )
        if "deepslot" in config.vocabulary:
            self.deepslot_classifier = MlpClassifier(
                input_size=embedding_size,
                hidden_size=config.deepslot_classifier_hidden_size,
                n_classes=len(config.vocabulary["deepslot"]),
                activation=config.activation,
                dropout=config.dropout
            )
        if "semclass" in config.vocabulary:
            self.semclass_classifier = MlpClassifier(
                input_size=embedding_size,
                hidden_size=config.semclass_classifier_hidden_size,
                n_classes=len(config.vocabulary["semclass"]),
                activation=config.activation,
                dropout=config.dropout
            )

    def forward(
        self,
        words: list[list[str]],
        counting_masks: LongTensor = None,
        lemma_rules: LongTensor = None,
        joint_feats: LongTensor = None,
        deps_ud: LongTensor = None,
        deps_eud: LongTensor = None,
        miscs: LongTensor = None,
        deepslots: LongTensor = None,
        semclasses: LongTensor = None,
        sent_ids: list[str] = None,
        texts: list[str] = None,
        inference_mode: bool = False
    ) -> CobaldParserOutput:
        result = {}

        # Extra [CLS] token accounts for the case when #NULL is the first token in a sentence.
        words_with_cls = prepend_cls(words)
        words_without_nulls = remove_nulls(words_with_cls)
        # Embeddings of words without nulls.
        embeddings_without_nulls = self.encoder(words_without_nulls)
        # Predict nulls.
        null_output = self.null_classifier(embeddings_without_nulls, counting_masks)
        result["counting_mask"] = null_output['preds']
        result["loss"] = null_output["loss"]

        # "Teacher forcing": during training, pass the original words (with gold nulls)
        # to the classification heads, so that they are trained upon correct sentences.
        if inference_mode:
            # Restore predicted nulls in the original sentences.
            result["words"] = add_nulls(words, null_output["preds"])
        else:
            result["words"] = words

        # Encode words with nulls.
        # [batch_size, seq_len, embedding_size]
        embeddings = self.encoder(result["words"])

        # Predict lemmas and morphological features.
        if hasattr(self, "lemma_rule_classifier"):
            lemma_output = self.lemma_rule_classifier(embeddings, lemma_rules)
            result["lemma_rules"] = lemma_output['preds']
            result["loss"] += lemma_output['loss']

        if hasattr(self, "morphology_classifier"):
            joint_feats_output = self.morphology_classifier(embeddings, joint_feats)
            result["joint_feats"] = joint_feats_output['preds']
            result["loss"] += joint_feats_output['loss']

        # Predict syntax.
        if hasattr(self, "dependency_classifier"):
            # [batch_size, seq_len]
            padding_mask = build_padding_mask(result["words"], self.device)
            null_mask = build_null_mask(result["words"], self.device)
            # Mask nulls for basic UD and don't mask for E-UD.
            deps_output = self.dependency_classifier(
                embeddings,
                deps_ud,
                deps_eud,
                mask_ud=(padding_mask & ~null_mask),
                mask_eud=padding_mask
            )
            result["deps_ud"] = deps_output['preds_ud']
            result["deps_eud"] = deps_output['preds_eud']
            result["loss"] += deps_output['loss_ud'] + deps_output['loss_eud']

        # Predict miscellaneous features.
        if hasattr(self, "misc_classifier"):
            misc_output = self.misc_classifier(embeddings, miscs)
            result["miscs"] = misc_output['preds']
            result["loss"] += misc_output['loss']

        # Predict semantics.
        if hasattr(self, "deepslot_classifier"):
            deepslot_output = self.deepslot_classifier(embeddings, deepslots)
            result["deepslots"] = deepslot_output['preds']
            result["loss"] += deepslot_output['loss']

        if hasattr(self, "semclass_classifier"):
            semclass_output = self.semclass_classifier(embeddings, semclasses)
            result["semclasses"] = semclass_output['preds']
            result["loss"] += semclass_output['loss']

        return CobaldParserOutput(**result)