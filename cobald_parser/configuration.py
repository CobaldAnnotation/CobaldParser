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
        # Vocabulary
        self.id2lemma_rule = id2lemma_rule
        self.id2morph_feats = id2morph_feats
        self.id2rel_ud = id2rel_ud
        self.id2rel_eud = id2rel_eud
        self.id2misc = id2misc
        self.id2deepslot = id2deepslot
        self.id2semclass = id2semclass
        super().__init__(**kwargs)