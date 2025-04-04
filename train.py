from copy import deepcopy # TODO: delete

from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer
)

from src.processing import preprocess, collate_with_padding
from src.parser import MorphoSyntaxSemanticsParserConfig, MorphoSyntaxSemanticsParser
from src.metrics import compute_metrics


def train(training_args: TrainingArguments, model_config_path: str):
    # FIXME
    val_dataset = load_dataset("CoBaLD/enhanced-cobald-dataset", name="en", trust_remote_code=True)['train'].take(100)
    # val_dataset.cleanup_cache_files()
    val_dataset = preprocess(val_dataset)

    # dataset = load_dataset("CoBaLD/enhanced-cobald-dataset", name="en", trust_remote_code=True)
    # dataset.cleanup_cache_files()
    train_dataset = deepcopy(val_dataset)
    # train_dataset = preprocess(train_dataset)

    # Load and autocomplete model config.
    model_config = MorphoSyntaxSemanticsParserConfig.from_json_file(model_config_path)
    # Do not list number of classes in configuration file.
    # Instead, automatically fill them at runtime for convenience.
    get_column_size = lambda column: train_dataset.features[column].feature.num_classes
    model_config.tagger_args["lemma_rule_classifier_args"]["n_classes"] = get_column_size("lemma_rules")
    model_config.tagger_args["morph_feats_classifier_args"]["n_classes"] = get_column_size("morph_feats")
    model_config.tagger_args["depencency_classifier_args"]["n_rels_ud"] = get_column_size("ud_deprels")
    model_config.tagger_args["depencency_classifier_args"]["n_rels_eud"] = get_column_size("eud_deprels")
    model_config.tagger_args["misc_classifier_args"]["n_classes"] = get_column_size("miscs")
    model_config.tagger_args["deepslot_classifier_args"]["n_classes"] = get_column_size("deepslots")
    model_config.tagger_args["semclass_classifier_args"]["n_classes"] = get_column_size("semclasses")
    model_config.null_predictor_args["consecutive_null_limit"] = get_column_size("counting_mask")
    # Create model.
    model = MorphoSyntaxSemanticsParser(model_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_with_padding,
        compute_metrics=compute_metrics,
    )
    trainer.train(ignore_keys_for_eval=['words', 'sent_id', 'text'])


if __name__ == "__main__":
    # Use HfArgumentParser with the built-in TrainingArguments class
    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument('--model_config', required=True)

    # Parse command-line arguments directly
    training_args, custom_args = parser.parse_args_into_dataclasses()
    
    train(training_args, custom_args.model_config)
