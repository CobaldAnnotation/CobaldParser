import os
from typing import override

from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import HfArgumentParser, TrainingArguments, Trainer
from transformers.modelcard import parse_log_history
from huggingface_hub import ModelCard, ModelCardData, EvalResult

from cobald_parser import CobaldParserConfig, CobaldParser
from src.processing import preprocess, collate_with_padding
from src.metrics import compute_metrics


MODELCARD_TEMPLATE = """
---
{{ card_data }}
---

# Model Card for {{ model_name }}

A transformer-based multihead parser for CoBaLD annotation.

This model parses a pre-tokenized CoNLL-U text and jointly labels each token with three tiers of tags:
* Grammatical tags (lemma, UPOS, XPOS, morphological features),
* Syntactic tags (basic and enhanced Universal Dependencies),
* Semantic tags (deep slot and semantic class).

## Model Sources

- **Repository:** https://github.com/CobaldAnnotation/CobaldParser
- **Paper:** [coming soon]
- **Demo:** [coming soon]

## Citation

[coming soon]
"""


class CustomTrainer(Trainer):
    @override
    def create_model_card(self, **kwargs):
        """Create custom model card."""

        dataset = self.eval_dataset
        organization, model_name = self.hub_model_id.split('/')
        hub_dataset_id = f"{organization}/{dataset.info.dataset_name}"

        _, _, eval_results = parse_log_history(self.state.log_history)

        def create_eval_result(metric_name: str, metric_type: str):
            return EvalResult(
                task_type='token-classification',
                dataset_type=hub_dataset_id,
                dataset_name=dataset.info.dataset_name,
                dataset_split='validation',
                metric_name=metric_name,
                metric_type=metric_type,
                metric_value=eval_results[metric_name]
            )

        card = ModelCard.from_template(
            card_data=ModelCardData(
                base_model=self.model.config.encoder_model_name,
                datasets=hub_dataset_id,
                language=dataset.info.config_name,
                eval_results=[
                    create_eval_result('Null F1', 'f1'),
                    create_eval_result('Lemma F1', 'f1'),
                    create_eval_result('Morphology F1', 'f1'),
                    create_eval_result('Ud Jaccard', 'accuracy'),
                    create_eval_result('Eud Jaccard', 'accuracy'),
                    create_eval_result('Miscs F1', 'f1'),
                    create_eval_result('Deepslot F1', 'f1'),
                    create_eval_result('Semclass F1', 'f1')
                ],
                library_name='transformers',
                license='gpl-3.0',
                metrics=['accuracy', 'f1'],
                model_name=self.hub_model_id,
                pipeline_tag='token-classification',
                tags=['pytorch']
            ),
            template_str=MODELCARD_TEMPLATE,
            model_name=model_name
        )
        model_card_filepath = os.path.join(self.args.output_dir, "README.md")
        card.save(model_card_filepath)


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
    parser.add_argument('--dataset_config_name', required=True)
    parser.add_argument('--model_config', required=True)

    # Parse command-line arguments.
    training_args, custom_args = parser.parse_args_into_dataclasses()

    dataset_dict = load_dataset(
        custom_args.dataset_path,
        name=custom_args.dataset_config_name,
        trust_remote_code=True
    )
    dataset_dict = preprocess(dataset_dict)

    # Print dataset information.
    print_dataset_info(dataset_dict)

    # Create and configure model.
    model_config = CobaldParserConfig.from_json_file(custom_args.model_config)
    model = CobaldParser(model_config)

    # Create trainer and train the model.
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict['train'].take(20),
        eval_dataset=dataset_dict['validation'].take(100),
        data_collator=collate_with_padding,
        compute_metrics=compute_metrics
    )
    trainer.train(ignore_keys_for_eval=['words', 'sent_id', 'text'])
    trainer.save_model()
