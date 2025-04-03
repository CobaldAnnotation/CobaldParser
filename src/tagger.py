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
        morph_feats_classifier_args: dict,
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
        self.morph_feats_classifier = MlpClassifier(
            input_size=embedding_size,
            **morph_feats_classifier_args,
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
        lemma_rules: LongTensor = None,
        morph_feats: LongTensor = None,
        syntax_ud: LongTensor = None,
        syntax_eud: LongTensor = None,
        miscs: LongTensor = None,
        deepslots: LongTensor = None,
        semclasses: LongTensor = None,
    ) -> dict[str, any]:

        # [batch_size, seq_len, embedding_size]
        embeddings = self.encoder(words)
        # [batch_size, seq_len]
        padding_mask = build_padding_mask(words, embeddings.device)
        null_mask = build_null_mask(words, embeddings.device)

        lemma_out = self.lemma_rule_classifier(embeddings, lemma_rules)
        morph_feats_out = self.morph_feats_classifier(embeddings, morph_feats)
        deps_out = self.dependency_classifier(
            embeddings,
            syntax_ud,
            syntax_eud,
            # Mask nulls for basic UD and don't mask for E-UD.
            mask_ud=(padding_mask & ~null_mask),
            mask_eud=padding_mask
        )
        misc_out = self.misc_classifier(embeddings, miscs)
        deepslot_out = self.deepslot_classifier(embeddings, deepslots)
        semclass_out = self.semclass_classifier(embeddings, semclasses)

        loss = (
            lemma_out['loss'] + morph_feats_out['loss'] +
            deps_out['loss_ud'] + deps_out['loss_eud'] + misc_out['loss'] +
            deepslot_out['loss'] + semclass_out['loss']
        )

        return {
            'lemma_rules': lemma_out['preds'],
            'morph_feats': morph_feats_out['preds'],
            'syntax_ud': deps_out['preds_ud'],
            'syntax_eud': deps_out['preds_eud'],
            'miscs': misc_out['preds'],
            'deepslots': deepslot_out['preds'],
            'semclasses': semclass_out['preds'],
            'loss': loss
        }
