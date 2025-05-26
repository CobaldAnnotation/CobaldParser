from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModel

from .configuration import CobaldParserConfig
from .modeling_parser import CobaldParser
from .pipeline import ConlluTokenClassificationPipeline

# Register model
CobaldParserConfig.register_for_auto_class()
CobaldParser.register_for_auto_class()

TASK_NAME = "conllu-parsing"
PIPELINE_REGISTRY.register_pipeline(
    task=TASK_NAME,
    pipeline_class=ConlluTokenClassificationPipeline,
    pt_model=AutoModel,
    type="text"
)