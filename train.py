import os
from typing import override

from torch.optim import AdamW
from datasets import load_dataset
from transformers import HfArgumentParser, TrainingArguments, Trainer, TrainerCallback
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
- **Paper:** https://dialogue-conf.org/wp-content/uploads/2025/04/BaiukIBaiukAPetrovaM.009.pdf
- **Demo:** [coming soon]

## Citation

@inproceedings{baiuk2025cobald,
  title={CoBaLD Parser: Joint Morphosyntactic and Semantic Annotation},
  author={Baiuk, Ilia and Baiuk, Alexandra and Petrova, Maria},
  booktitle={Proceedings of the International Conference "Dialogue"},
  volume={I},
  year={2025}
}
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
                pipeline_tag='cobald-parsing', # Use the correct task name
                tags=['pytorch']
            ),
            template_str=MODELCARD_TEMPLATE,
            model_name=model_name
        )
        model_card_filepath = os.path.join(self.args.output_dir, "README.md")
        card.save(model_card_filepath)

    @override
    def create_optimizer(self):
        # Implement discriminative‐finetuning.
        # NOTE: it breaks multiple CLI features like `--fp16` and `--fsdp`, but
        # we don't need them so far anyway...

        if self.optimizer is not None:
            return self.optimizer
        
        base_lr = self.args.learning_rate
        encoder_lr = base_lr / 5
        decay = self.args.weight_decay
        layer_decay = 0.9
        optimizer_grouped_parameters = []

        # Add classifier with the base LR
        classifiers_params = [
            *self.model.null_classifier.parameters(),
            *self.model.lemma_rule_classifier.parameters(),
            *self.model.morph_feats_classifier.parameters(),
            *self.model.dependency_classifier.parameters(),
            *self.model.misc_classifier.parameters(),
            *self.model.deepslot_classifier.parameters(),
            *self.model.semclass_classifier.parameters()
        ]
        optimizer_grouped_parameters.append({
            "params": classifiers_params,
            "lr": base_lr,
            "weight_decay": decay
        })
        
        # Per‐layer parameter groups with decaying LR
        layers = self.model.encoder.get_transformer_layers()
        for idx, layer in enumerate(layers):
            lr = encoder_lr * (layer_decay ** (len(layers) - idx - 1))
            optimizer_grouped_parameters.append({
                "params": layer.parameters(),
                "lr": lr,
                "weight_decay": decay
            })

        # Add embeddings with the smallest LR
        embeddings = self.model.encoder.get_embeddings()
        smallest_lr = encoder_lr * (layer_decay ** len(layers))
        optimizer_grouped_parameters.append({
            "params": embeddings.parameters(),
            "lr": smallest_lr,
            "weight_decay": decay
        })

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon
        )
        return self.optimizer


class GradualUnfreezeCallback(TrainerCallback):
    """Unfreeze one encoder layer per epoch, deepest first."""

    def __init__(self, warmup: int = 1, interval: int = 3):
        self.warmup = warmup
        self.interval = interval

    def on_train_begin(self, args, state, control, model = None, **kwargs):
        # Freeze encoder at start
        for param in model.encoder.parameters():
            param.requires_grad = False

    def on_epoch_begin(self, args, state, control, model = None, **kwargs):
        epoch = int(state.epoch)

        # Keep encoder frozen during warmup
        if epoch < self.warmup:
            return
        
        layers = model.encoder.get_transformer_layers()
        top_layer_idx = len(layers) - 1
        last_frozen_layer_idx = top_layer_idx - epoch * self.interval

        # Gradually unfreeze layers from top to bottom or unfreeze encoder entirely
        # (e.g. including the embeddings) if all layers are already unfreezed.
        if last_frozen_layer_idx < 0:
            for param in model.encoder.parameters():
                param.requires_grad = True
        else:
            for layer in layers[top_layer_idx:last_frozen_layer_idx:-1]:
                for param in layer.parameters():
                    param.requires_grad = True


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
        name=custom_args.dataset_config_name
    )
    dataset_dict = preprocess(dataset_dict)

    # Create and configure model.
    model_config = CobaldParserConfig.from_json_file(custom_args.model_config)
    # Map labels and to ids based on dataset features.
    id2label_kwargs = {}
    for config_arg, feature_name in [
        ("id2lemma_rule", "lemma_rules"),
        ("id2morph_feats", "morph_feats"),
        ("id2rel_ud", "ud_deprels"),
        ("id2rel_eud", "eud_deprels"),
        ("id2misc", "miscs"),
        ("id2deepslot", "deepslots"),
        ("id2semclass", "semclasses")
    ]:
        feature = dataset_dict['train'].features[feature_name].feature
        id2label_kwargs[config_arg] = dict(enumerate(feature.names))
    model_config.update(id2label_kwargs)

    model = CobaldParser(model_config)

    # Create trainer and train the model.
    unfreeze_callback = GradualUnfreezeCallback()
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['validation'],
        data_collator=collate_with_padding,
        compute_metrics=compute_metrics,
        callbacks=[unfreeze_callback]
    )
    trainer.train(ignore_keys_for_eval=['words', 'sent_id', 'text'])
    # Save and push model to hub (if push_to_hub is set).
    trainer.save_model()