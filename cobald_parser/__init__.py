from transformers import AutoConfig, AutoModelForTokenClassification

from .configuration import CobaldParserConfig
from .modeling_parser import CobaldParser

AutoConfig.register("cobald_parser", CobaldParserConfig)
AutoModelForTokenClassification.register(CobaldParserConfig, CobaldParser)