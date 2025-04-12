from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer
)

from cobald_parser import CobaldParserConfig, CobaldParser
from src.processing import preprocess, collate_with_padding
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


if __name__ == "__main__":
    # Use HfArgumentParser with the built-in TrainingArguments class
    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--model_config', required=True)

    # Parse command-line arguments.
    training_args, custom_args = parser.parse_args_into_dataclasses()

    dataset_dict = load_dataset(
        custom_args.dataset_path,
        name=custom_args.dataset_name,
        trust_remote_code=True
    )
    dataset_dict = preprocess(dataset_dict)

    # Print dataset information.
    print_dataset_info(dataset_dict)

    # Create and configure model.
    model_config = CobaldParserConfig.from_json_file(custom_args.model_config)
    model = CobaldParser(model_config)

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
    trainer.save_model()
