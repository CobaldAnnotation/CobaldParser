from transformers.pipelines import PIPELINE_REGISTRY

from .configuration import CobaldParserConfig
from .modeling_parser import CobaldParser
from .pipeline import ConlluTokenClassificationPipeline

# Register model
CobaldParserConfig.register_for_auto_class()
CobaldParser.register_for_auto_class()

PIPELINE_REGISTRY.register_pipeline(
    task="conllu-parsing",
    pipeline_class=ConlluTokenClassificationPipeline,
    pt_model=CobaldParser,
    type="text"
)