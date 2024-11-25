from torch import nn
from torch import Tensor

from null_predictor import NullPredictor
from tagger import MultiHeadTagger
from encoder import MaskedLanguageModelEncoder


class MorphoSyntaxSemanticsParser(nn.Module):
    """Morpho-Syntax-Semantic Parser."""

    def __init__(
        self,
        encoder_args: dict,
        null_predictor_args: dict,
        tagger_args: dict
    ):
        super().__init__()

        encoder = MaskedLanguageModelEncoder(**encoder_args)
        self.null_predictor = NullPredictor(encoder=encoder, **null_predictor_args)
        self.tagger = MultiHeadTagger(encoder=encoder, **tagger_args)

    def forward(
        self,
        words: list[list[str]],
        lemma_rules: Tensor = None,
        joint_pos_feats: Tensor = None,
        deps_ud: Tensor = None,
        deps_eud: Tensor = None,
        miscs: Tensor = None,
        deepslots: Tensor = None,
        semclasses: Tensor = None
    ) -> dict[str, any]:

        # If no labels for any of three tiers are provided, we are at inference.
        has_labels = lemma_rules is not None or joint_pos_feats is not None \
            or deps_ud is not None or deps_eud is not None or miscs is not None \
            or deepslots is not None or semclasses is not None
        # Restore nulls.
        null_out = self.null_predictor(words, is_inference=has_labels)
        # Words with predicted nulls.
        words_with_nulls = null_out['words']

        # Teacher forcing: during training, pass the original words (with gold nulls)
        # to the tagger, so that the latter is trained upon correct sentences.
        # Moreover, we cannot calculate loss on predicted nulls, as they have no labels,
        # so the same strategy is used for validation as well.
        if has_labels:
            words_with_nulls = words

        # Predict morphological, syntactic and semantic tags.
        tagger_out = self.tagger(
            words_with_nulls,
            lemma_rules,
            joint_pos_feats,
            deps_ud,
            deps_eud,
            miscs,
            deepslots,
            semclasses
        )

        # Add up null predictor and tagger losses.
        loss = null_out['loss'] + tagger_out['loss']

        return {
            'words': null_out['words'],
            'lemma_rules': tagger_out['lemma_rules'],
            'joint_pos_feats': tagger_out['joint_pos_feats'],
            'deps_ud': tagger_out['deps_ud'],
            'deps_eud': tagger_out['deps_eud'],
            'miscs': tagger_out['miscs'],
            'deepslots': tagger_out['deepslots'],
            'semclasses': tagger_out['semclasses'],
            'loss': loss
        }

