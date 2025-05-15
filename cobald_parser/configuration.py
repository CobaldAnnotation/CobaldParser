from transformers import PretrainedConfig


class CobaldParserConfig(PretrainedConfig):
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
        activation: str = 'relu',
        dropout: float = 0.1,
        consecutive_null_limit: int = 0,
        id2lemma_rule: dict[int, str] = {},
        id2morph_feats: dict[int, str] = {},
        id2rel_ud: dict[int, str] = {},
        id2rel_eud: dict[int, str] = {},
        id2misc: dict[int, str] = {},
        id2deepslot: dict[int, str] = {},
        id2semclass: dict[int, str] = {},
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
        self.activation = activation
        self.dropout = dropout
        # Vocabulary.
        # The config stores mappings as strings, so we have to convert them to int.
        cast_keys_to_int = lambda id2label: {int(k): v for k, v in id2label.items()}
        self.id2lemma_rule = cast_keys_to_int(id2lemma_rule)
        self.id2morph_feats = cast_keys_to_int(id2morph_feats)
        self.id2rel_ud = cast_keys_to_int(id2rel_ud)
        self.id2rel_eud = cast_keys_to_int(id2rel_eud)
        self.id2misc = cast_keys_to_int(id2misc)
        self.id2deepslot = cast_keys_to_int(id2deepslot)
        self.id2semclass = cast_keys_to_int(id2semclass)
        # HACK: Tell HF hub about custom pipeline.
        # It should not be hardcoded like this but other workaround are worse imo.
        self.custom_pipelines = {
            "cobald-parsing": {
                "impl": "pipeline.ConlluTokenClassificationPipeline",
                "pt": "CobaldParser",
            }
        }
        super().__init__(**kwargs)