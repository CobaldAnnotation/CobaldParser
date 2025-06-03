from collections import defaultdict

from datasets import load_dataset, concatenate_datasets
from transformers import (
    HfArgumentParser,
    TrainingArguments
)

from cobald_parser import (
    CobaldParserConfig,
    CobaldParser,
    ConlluTokenClassificationPipeline
)
from src.processing import (
    transform_dataset,
    extract_unique_labels,
    build_schema_with_class_labels,
    replace_none_with_ignore_index,
    collate_with_padding,
    LEMMA_RULE,
    JOINT_FEATS,
    UD_DEPREL,
    EUD_DEPREL,
    MISC,
    SEMCLASS,
    DEEPSLOT
)
from src.callbacks import GradualUnfreezeCallback
from src.trainer import CustomTrainer
from src.metrics import compute_metrics


def parse_datasets(value: str) -> list[tuple]:
    result = []
    datasets_configs = value.split(',')
    for dataset_config in datasets_configs:
        parts = dataset_config.split(':')
        if len(parts) != 2:
            raise ValueError(f"Dataset '{value}' is not in the format 'name:config'")
        dataset, config = parts
        result.append((dataset, config))
    return result


def build_shared_tagsets(datasets_configs: list[tuple], allowed_columns: set = None) -> dict:
    tagsets = defaultdict(set)
    for dataset_name, config_name in datasets_configs:
        external_dataset_dict = load_dataset(dataset_name, name=config_name)
        external_dataset_dict = transform_dataset(external_dataset_dict)
        external_dataset = concatenate_datasets(external_dataset_dict.values())
        for column_name in external_dataset.column_names:
            # Skip columns that are not marked as allowed
            if allowed_columns is not None and column_name not in allowed_columns:
                continue
            tagsets[column_name] |= extract_unique_labels(external_dataset, column_name)
    return tagsets


def update_vocabulary(config, features):
    for column in [LEMMA_RULE, JOINT_FEATS, UD_DEPREL, EUD_DEPREL, MISC, DEEPSLOT, SEMCLASS]:
        if column in features:
            labels = features[column].feature.names
            config.vocabulary[column] = dict(enumerate(labels))


def transfer_pretrained(model, pretrained_model):
    if not isinstance(pretrained_model, CobaldParser):
        raise ValueError(f"Pretrained model must be CobaldParser class instance")

    # Transfer encoder
    model.encoder = pretrained_model.encoder

    a = set(model.config.vocabulary[EUD_DEPREL].items())
    b = set(pretrained_model.config.vocabulary[EUD_DEPREL].items())

    print(f"diff: {a - b}")

    # Transfer classifiers
    for name in model.classifiers:
        if name in pretrained_model.classifiers:
            try:
                # Try to transfer weights from pretrained classifier if it matches
                # the shape of the model's classifier (e.g. hidden_size, n_classes, etc.)
                pretrained_classifier_state = pretrained_model.classifiers[name].state_dict()
                model.classifiers[name].load_state_dict(pretrained_classifier_state)
                print(f"Successfuly transfered {name} classifier")
            except Exception as e:
                print(f"Could not transfer {name} classifier:\n{e}")


if __name__ == "__main__":
    # Use HfArgumentParser with the built-in TrainingArguments class
    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument('--model_config', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--dataset_config_name')
    parser.add_argument(
        '--external_datasets',
        type=parse_datasets,
        nargs="?",
        help="External datasets whose colums will be added to the model's vocabulary "
        "in format `dataset1_path:config1_name,dataset2_path:config2_name,...` "
        "(use `default` if no other configs exist). "
        "Example: `CoBaLD/enhanced-cobald:en,CoBaLD/enhanced-ud-syntax:default`."
    )
    parser.add_argument('--finetune_from')

    # Parse command-line arguments.
    training_args, custom_args = parser.parse_args_into_dataclasses()

    target_dataset_dict = load_dataset(
        custom_args.dataset_path,
        name=custom_args.dataset_config_name
    )
    target_dataset_dict = transform_dataset(target_dataset_dict)

    all_datasets = [(custom_args.dataset_path, custom_args.dataset_config_name)]
    if custom_args.external_datasets:
        all_datasets.extend(custom_args.external_datasets)

    tagsets = build_shared_tagsets(
        all_datasets,
        allowed_columns=target_dataset_dict['train'].column_names
    )
    schema = build_schema_with_class_labels(tagsets)

    # Final processing.
    target_dataset_dict = (
        target_dataset_dict
        .cast(schema)
        .map(replace_none_with_ignore_index)
        .with_format("torch")
    )

    # Create and configure model.
    model_config = CobaldParserConfig.from_json_file(custom_args.model_config)
    # Load vocabulary into config (as it must be saved along the model).
    update_vocabulary(model_config, target_dataset_dict['train'].features)

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

    if custom_args.finetune_from:
        pretrained_model = CobaldParser.from_pretrained(
            custom_args.finetune_from,
            trust_remote_code=True
        )
        transfer_pretrained(model, pretrained_model)

    # Create trainer and train the model.
    unfreeze_callback = GradualUnfreezeCallback()
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=target_dataset_dict['train'],
        eval_dataset=target_dataset_dict['validation'],
        data_collator=collate_with_padding,
        # Wth? See notes at compute_metrics.
        compute_metrics=lambda x: compute_metrics(x, training_args.label_names),
        callbacks=[unfreeze_callback]
    )
    trainer.train(ignore_keys_for_eval=["words", "sent_ids", "texts"])

    # Save and push model to hub (if push_to_hub is set).
    trainer.save_model()

    pipe = ConlluTokenClassificationPipeline(model)
    pipe.push_to_hub(training_args.hub_model_id)