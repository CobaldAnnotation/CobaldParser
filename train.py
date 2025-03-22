from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer
)
from src.processing import preprocess, collate_with_ignore_index
from src.parser import MorphoSyntaxSemanticsParserConfig, MorphoSyntaxSemanticsParser

# FIXME
import importlib
import transformers
importlib.reload(transformers)

# TODO
def compute_metrics(eval_pred, compute_result: bool):
    preds, labels = eval_pred
    print(f"compute_metrics is called")
    # print(preds)
    # print(labels)
    print(f"preds: {len(preds)} items: {list(pred.shape for pred in preds)}")
    print(f"labels: {len(labels)} items: {list(label.shape for label in labels)}")
    return {"eval_loss": 0.0}


def train(training_args: TrainingArguments, model_config_path: str):
    dataset = load_dataset("CoBaLD/enhanced-cobald-dataset", name="en", trust_remote_code=True)
    dataset.cleanup_cache_files()
    train_dataset = dataset['train']
    train_dataset = preprocess(train_dataset)

    # FIXME
    val_dataset = load_dataset("CoBaLD/enhanced-cobald-dataset", name="en", trust_remote_code=True)['train']
    val_dataset.cleanup_cache_files()
    val_dataset = preprocess(val_dataset)

    # Create model.
    model_config = MorphoSyntaxSemanticsParserConfig.from_json_file(model_config_path)
    get_column_size = lambda column: train_dataset.features[column].feature.num_classes
    get_matrix_column_size = lambda column: train_dataset.features[column].feature.feature.num_classes
    # Do not list number of classes in configuration file.
    # Instead, automatically fill them at runtime for convenience.
    model_config.tagger_args["lemma_rule_classifier_args"]["n_classes"] = get_column_size("lemma_rules")
    model_config.tagger_args["morph_feats_classifier_args"]["n_classes"] = get_column_size("morph_feats")
    model_config.tagger_args["depencency_classifier_args"]["n_rels_ud"] = get_matrix_column_size("syntax_ud")
    model_config.tagger_args["depencency_classifier_args"]["n_rels_eud"] = get_matrix_column_size("syntax_eud")
    model_config.tagger_args["misc_classifier_args"]["n_classes"] = get_column_size("miscs")
    model_config.tagger_args["deepslot_classifier_args"]["n_classes"] = get_column_size("deepslots")
    model_config.tagger_args["semclass_classifier_args"]["n_classes"] = get_column_size("semclasses")
    model_config.null_predictor_args["consecutive_null_limit"] = get_column_size("counting_mask")
    model = MorphoSyntaxSemanticsParser(model_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_with_ignore_index,
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
