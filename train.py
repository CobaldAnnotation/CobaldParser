from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    PreTrainedModel
)

from src.processing import preprocess, collate_with_padding
from src.parser import MorphoSyntaxSemanticsParserConfig, MorphoSyntaxSemanticsParser
from src.metrics import compute_metrics


def print_dataset_info(dataset_dict: DatasetDict):
    print("\nDataset Information:")
    print(f"{'Column Name':<30} {'n_classes'}")
    full_dataset = concatenate_datasets(dataset_dict.values())
    for column in full_dataset.features:
        try:
            print(f"{column:<30} {full_dataset.features[column].feature.num_classes}")
        except:
            pass
    print('-----------------------')


def configure_model(model_config_path: str, pretrained_model_path: str = None) -> PreTrainedModel:
    # Load model config
    model_config = MorphoSyntaxSemanticsParserConfig.from_json_file(model_config_path)
    
    # Create model or load pretrained one for fine-tuning
    if pretrained_model_path:
        model = MorphoSyntaxSemanticsParser.from_pretrained(
            pretrained_model_path,
            config=model_config
        )
    else:
        model = MorphoSyntaxSemanticsParser(model_config)

    return model


if __name__ == "__main__":
    # Use HfArgumentParser with the built-in TrainingArguments class
    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument('--model_config', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--pretrained_model_path', default=None,
                        help="Path to pretrained model for fine-tuning")

    # Parse command-line arguments directly
    training_args, custom_args = parser.parse_args_into_dataclasses()

    dataset_dict = load_dataset(
        custom_args.dataset_path,
        name=custom_args.dataset_name,
        trust_remote_code=True
    )
    dataset_dict = preprocess(dataset_dict)

    # Print dataset information
    print_dataset_info(dataset_dict)

    # Create and configure model.
    model = configure_model(custom_args.model_config, custom_args.pretrained_model_path)

    # Create trainer and train the model.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['validation'],
        data_collator=collate_with_padding,
        compute_metrics=compute_metrics
    )
    trainer.train(ignore_keys_for_eval=['words', 'sent_id', 'text'])
