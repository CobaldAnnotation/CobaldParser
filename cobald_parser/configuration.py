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