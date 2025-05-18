import os
from typing import override

from torch.optim import AdamW
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from transformers.modelcard import parse_log_history
from huggingface_hub import ModelCard, ModelCardData, EvalResult

from cobald_parser import CobaldParserConfig, CobaldParser
from src.processing import (
    preprocess,
    collate_with_padding,
    COUNTING_MASK,
    LEMMA_RULE,
    JOINT_FEATS,
    UD_DEPREL,
    EUD_DEPREL,
    MISC,
    SEMCLASS,
    DEEPSLOT
)
from src.metrics import compute_metrics


def export_vocabulary(train_dataset_features, config):
    for column in [LEMMA_RULE, JOINT_FEATS, UD_DEPREL, EUD_DEPREL, MISC, DEEPSLOT, SEMCLASS]:
        if column in train_dataset_features:
            labels = train_dataset_features[column].feature.names
            config.vocabulary[column] = dict(enumerate(labels))


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
        classifiers_params = []
        classifiers_params.extend(self.model.null_classifier.parameters())
        if hasattr(self.model, "lemma_rule_classifier"):
            classifiers_params.extend(self.model.lemma_rule_classifier.parameters())
        if hasattr(self.model, "morphology_classifier"):
            classifiers_params.extend(self.model.morphology_classifier.parameters())
        if hasattr(self.model, "dependency_classifier"):
            classifiers_params.extend(self.model.dependency_classifier.parameters())
        if hasattr(self.model, "misc_classifier"):
            classifiers_params.extend(self.model.misc_classifier.parameters())
        if hasattr(self.model, "deepslot_classifier"):
            classifiers_params.extend(self.model.deepslot_classifier.parameters())
        if hasattr(self.model, "semclass_classifier"):
            classifiers_params.extend(self.model.semclass_classifier.parameters())
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
        embeddings = self.model.encoder.get_embeddings_layer()
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
    parser.add_argument('--model_config', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--dataset_config_name')

    # Parse command-line arguments.
    training_args, custom_args = parser.parse_args_into_dataclasses()

    dataset_dict = load_dataset(
        custom_args.dataset_path,
        name=custom_args.dataset_config_name
    )
    dataset_dict = preprocess(dataset_dict)

    # Create and configure model.
    model_config = CobaldParserConfig.from_json_file(custom_args.model_config)
    # Export vocabulary to config (as it must be saved along the model).
    export_vocabulary(dataset_dict['train'].features, model_config)

    # Manually set some parameters for this specific workflow to work.
    training_args.remove_unused_columns = False
    training_args.label_names = ["counting_masks"]
    for dataset_column, parser_input in (
        (LEMMA_RULE, "lemma_rules"),
        (JOINT_FEATS, "joint_feats"),
        (UD_DEPREL, "deps_ud"),
        (EUD_DEPREL, "deps_eud"),
        (MISC, "miscs"),
        (DEEPSLOT, "deepslots"),
        (SEMCLASS, "semclasses")
    ):
        if dataset_column in model_config.vocabulary:
            training_args.label_names.append(parser_input)

    model = CobaldParser(model_config)

    # Create trainer and train the model.
    unfreeze_callback = GradualUnfreezeCallback()
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['validation'],
        data_collator=collate_with_padding,
        # Wth? See notes at compute_metrics.
        compute_metrics=lambda x: compute_metrics(x, training_args.label_names),
        callbacks=[unfreeze_callback]
    )
    trainer.train(ignore_keys_for_eval=["words", "sent_ids", "texts"])
    # Save and push model to hub (if push_to_hub is set).
    trainer.save_model()