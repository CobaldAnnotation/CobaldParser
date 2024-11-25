from torch import nn
from torch import tensor

from mlp_classifier import MLPClassifier
from dependency_classifier import DependencyClassifier
from encoder import MaskedLanguageModelEncoder
from utils import build_padding_mask, build_null_mask


class MultiHeadTagger(nn.Module):
    """Morpho-Syntax-Semantic tagger."""

    def __init__(
        self,
        encoder: MaskedLanguageModelEncoder,
        lemma_rule_classifier_args: dict,
        pos_feats_classifier_args: dict,
        depencency_classifier_args: dict,
        misc_classifier_args: dict,
        deepslot_classifier_args: dict,
        semclass_classifier_args: dict
    ):
        super().__init__()

        self.encoder = encoder
        embedding_size = self.encoder.get_embedding_size()

        # Heads.
        self.lemma_rule_classifier = MLPClassifier(
            input_size=embedding_size,
            **lemma_rule_classifier_args
        )
        self.joint_pos_feats_classifier = MLPClassifier(
            input_size=embedding_size,
            **pos_feats_classifier_args
        )
        self.dependency_classifier = DependencyClassifier(
            input_size=embedding_size,
            **depencency_classifier_args
        )
        self.misc_classifier = MLPClassifier(
            input_size=embedding_size,
            **misc_classifier_args
        )
        self.deepslot_classifier = MLPClassifier(
            input_size=embedding_size,
            **deepslot_classifier_args
        )
        self.semclass_classifier = MLPClassifier(
            input_size=embedding_size,
            **semclass_classifier_args
        )

    def forward(
        self,
        words: list[list[str]],
        lemma_rules: tensor = None,
        joint_pos_feats: tensor = None,
        deps_ud: tensor = None,
        deps_eud: tensor = None,
        miscs: tensor = None,
        deepslots: tensor = None,
        semclasses: tensor = None
    ) -> dict[str, any]:

        # [batch_size, seq_len, embedding_size]
        embeddings = self.encoder(words)
        # [batch_size, seq_len]
        padding_mask = build_padding_mask(words, embeddings.device)
        null_mask = build_null_mask(words, embeddings.device)

        lemma_out = self.lemma_rule_classifier(embeddings, lemma_rules, padding_mask)
        joint_pos_feats_out = self.joint_pos_feats_classifier(embeddings, joint_pos_feats, padding_mask)
        deps_out = self.dependency_classifier(
            embeddings,
            deps_ud,
            deps_eud,
            # Mask nulls for basic UD and don't mask for E-UD.
            mask_ud=(padding_mask & ~null_mask),
            mask_eud=padding_mask
        )
        misc_out = self.misc_classifier(embeddings, miscs, padding_mask)
        deepslot_out = self.deepslot_classifier(embeddings, deepslots, padding_mask)
        semclass_out = self.semclass_classifier(embeddings, semclasses, padding_mask)

        loss = lemma_out['loss'] \
            + joint_pos_feats_out['loss'] \
            + deps_out['loss_ud'] \
            + deps_out['loss_eud'] \
            + misc_out['loss'] \
            + deepslot_out['loss'] \
            + semclass_out['loss']

        return {
            'lemma_rules': lemma_out['preds'],
            'joint_pos_feats': joint_pos_feats_out['preds'],
            'deps_ud': deps_out['preds_ud'],
            'deps_eud': deps_out['preds_eud'],
            'miscs': misc_out['preds'],
            'deepslots': deepslot_out['preds'],
            'semclasses': semclass_out['preds'],
            'loss': loss
        }

