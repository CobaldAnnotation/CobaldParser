from torch import nn
from torch import LongTensor

from src.mlp_classifier import MlpClassifier
from src.dependency_classifier import DependencyClassifier
from src.encoder import MaskedLanguageModelEncoder
from src.utils import build_padding_mask, build_null_mask


class MultiHeadTagger(nn.Module):
    """Morpho-Syntax-Semantic tagger."""

    def __init__(
        self,
        encoder: MaskedLanguageModelEncoder,
        lemma_rule_classifier_args: dict,
        joint_pos_feats_classifier_args: dict,
        depencency_classifier_args: dict,
        misc_classifier_args: dict,
        deepslot_classifier_args: dict,
        semclass_classifier_args: dict
    ):
        super().__init__()

        self.encoder = encoder
        embedding_size = self.encoder.get_embedding_size()

        # Heads.
        self.lemma_rule_classifier = MlpClassifier(
            input_size=embedding_size,
            **lemma_rule_classifier_args,
        )
        self.joint_pos_feats_classifier = MlpClassifier(
            input_size=embedding_size,
            **joint_pos_feats_classifier_args,
        )
        self.dependency_classifier = DependencyClassifier(
            input_size=embedding_size,
            **depencency_classifier_args,
        )
        self.misc_classifier = MlpClassifier(
            input_size=embedding_size,
            **misc_classifier_args,
        )
        self.deepslot_classifier = MlpClassifier(
            input_size=embedding_size,
            **deepslot_classifier_args,
        )
        self.semclass_classifier = MlpClassifier(
            input_size=embedding_size,
            **semclass_classifier_args,
        )

    def forward(
        self,
        words: list[list[str]],
        lemma_rule_labels: LongTensor = None,
        joint_pos_feats_labels: LongTensor = None,
        deps_ud_labels: LongTensor = None,
        deps_eud_labels: LongTensor = None,
        misc_labels: LongTensor = None,
        deepslot_labels: LongTensor = None,
        semclass_labels: LongTensor = None,
    ) -> dict[str, any]:

        # [batch_size, seq_len, embedding_size]
        embeddings = self.encoder(words)
        # [batch_size, seq_len]
        padding_mask = build_padding_mask(words, embeddings.device)
        null_mask = build_null_mask(words, embeddings.device)

        lemma_out = self.lemma_rule_classifier(
            embeddings,
            lemma_rule_labels,
            padding_mask
        )
        joint_pos_feats_out = self.joint_pos_feats_classifier(
            embeddings,
            joint_pos_feats_labels,
            padding_mask
        )
        deps_out = self.dependency_classifier(
            embeddings,
            deps_ud_labels,
            deps_eud_labels,
            # Mask nulls for basic UD and don't mask for E-UD.
            mask_ud=(padding_mask & ~null_mask),
            mask_eud=padding_mask
        )
        misc_out = self.misc_classifier(embeddings, misc_labels, padding_mask)
        deepslot_out = self.deepslot_classifier(
            embeddings,
            deepslot_labels,
            padding_mask
        )
        semclass_out = self.semclass_classifier(
            embeddings,
            semclass_labels,
            padding_mask
        )

        loss = lemma_out['loss'] \
            + joint_pos_feats_out['loss'] \
            + deps_out['loss_ud'] \
            + deps_out['loss_eud'] \
            + misc_out['loss'] \
            + deepslot_out['loss'] \
            + semclass_out['loss']

        return {
            'lemma_rule_preds': lemma_out['preds'],
            'joint_pos_feats_preds': joint_pos_feats_out['preds'],
            'deps_ud_preds': deps_out['preds_ud'],
            'deps_eud_preds': deps_out['preds_eud'],
            'misc_preds': misc_out['preds'],
            'deepslot_preds': deepslot_out['preds'],
            'semclass_preds': semclass_out['preds'],
            'loss': loss
        }
