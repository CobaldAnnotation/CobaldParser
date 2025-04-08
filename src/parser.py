from torch import LongTensor

from transformers import PretrainedConfig, PreTrainedModel

from src.null_predictor import NullPredictor
from src.tagger import MultiHeadTagger
from src.encoder import MaskedLanguageModelEncoder


class MorphoSyntaxSemanticsParserConfig(PretrainedConfig):
    model_type = "cobald_parser"

    def __init__(
        self,
        encoder_args: dict = {},
        null_predictor_args: dict = {},
        tagger_args: dict = {},
        **kwargs,
    ):
        # NOTE: huggingface nested configs serialization is a mess
        # that doesn't work without explicitly overriding the to_dict/from_dict methods
        # to properly reinstantiate nested configuration objects from their
        # dict representations.
        # That's why we use dicts instead of sub-configs here.
        self.encoder_args = encoder_args
        self.null_predictor_args = null_predictor_args
        self.tagger_args = tagger_args
        super().__init__(**kwargs)


class MorphoSyntaxSemanticsParser(PreTrainedModel):
    """Morpho-Syntax-Semantic Parser."""

    def __init__(self, config: MorphoSyntaxSemanticsParserConfig):
        super().__init__(config)
        encoder = MaskedLanguageModelEncoder(**config.encoder_args)
        self.null_predictor = NullPredictor(
            encoder=encoder,
            **config.null_predictor_args
        )
        self.tagger = MultiHeadTagger(
            encoder=encoder,
            **config.tagger_args
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
        
        # Restore nulls.
        null_output = self.null_predictor(words, counting_mask)

        # "Teacher forcing": during training, pass the original words (with gold nulls)
        # to the tagger, so that the latter is trained upon correct sentences.
        if inference_mode:
            words_with_nulls = null_output['words_with_nulls']
        else:
            words_with_nulls = words

        # Predict morphological, syntactic and semantic tags.
        tagger_output = self.tagger(
            words_with_nulls,
            lemma_rules,
            morph_feats,
            syntax_ud,
            syntax_eud,
            miscs,
            deepslots,
            semclasses
        )

        # Add up null predictor and tagger losses.
        loss = null_output['loss'] + tagger_output['loss']

        return {
            'words': null_output['words_with_nulls'],
            'counting_mask': null_output['counting_mask'],
            'lemma_rules': tagger_output['lemma_rules'],
            'morph_feats': tagger_output['morph_feats'],
            'syntax_ud': tagger_output['syntax_ud'],
            'syntax_eud': tagger_output['syntax_eud'],
            'miscs': tagger_output['miscs'],
            'deepslots': tagger_output['deepslots'],
            'semclasses': tagger_output['semclasses'],
            'loss': loss
        }
