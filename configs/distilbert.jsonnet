# Configuration file for baseline model pretraining.
# See https://guide.allennlp.org/training-and-prediction#2 for guidance.
{
    "train_data_path": "data/train.conllu",
    "validation_data_path": "data/validation.conllu",
    "dataset_reader": {
        "type": "compreno_ud_dataset_reader", # Use custom dataset reader.
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": "distilbert-base-uncased"
            }
        }
    },
    "data_loader": {
        "batch_size": 16,
        "shuffle": true
    },
    "validation_data_loader": {
        "batch_size": 16,
        "shuffle": false
    },
    # Extend target vocabulary with tags encountered in pretraining dataset.
    "vocabulary": {
        "min_count": {
            # Ignore lemmatization rules encountered 1 time in training dataset.
            "lemma_rule_labels": 2
        },
        "tokens_to_add": { # Add default OOV tokens.
            "lemma_rule_labels": ["@@UNKNOWN@@"],
        }
    },
    "model": {
        "type": "morpho_syntax_semantic_parser", # Use custom model.
        # FIXME: take indexer from dataset_reader
        "indexer": {
            "type": "pretrained_transformer_mismatched",
            "model_name": "distilbert-base-uncased"
        },
        "embedder": {
            "type": "pretrained_transformer_mismatched",
            "model_name": "distilbert-base-uncased",
            "train_parameters": true
        },
        "lemma_rule_classifier": {
            "hid_dim": 512,
            "activation": "relu",
            "dropout": 0.1,
        },
        "pos_feats_classifier": {
            "hid_dim": 256,
            "activation": "relu",
            "dropout": 0.1
        },
        "depencency_classifier": {
            "hid_dim": 128,
            "activation": "relu",
            "dropout": 0.1
        },
        "misc_classifier": {
            "hid_dim": 128,
            "activation": "relu",
            "dropout": 0.1
        },
        "semslot_classifier": {
            "hid_dim": 1024,
            "activation": "relu",
            "dropout": 0.1
        },
        "semclass_classifier": {
            "hid_dim": 1024,
            "activation": "relu",
            "dropout": 0.1
        },
        "null_classifier": {
            "hid_dim": 512,
            "activation": "relu",
            "dropout": 0.1
        }
    },
    "trainer": {
        "type": "gradient_descent",
        "optimizer": {
            "type": "adam",
            "lr": 1e-2, # Base (largest) learning rate.
            "parameter_groups": [
                [ # Second group of layers.
                    ["embedder"], {}
                ],
                [ # First group of layers.
                    [
                    "lemma_rule_classifier",
                    "pos_feats_classifier",
                    "dependency_classifier",
                    "misc_classifier",
                    "semslot_classifier",
                    "semclass_classifier",
                    "null_classifier"
                    ], {}
                ],
            ],
        },
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "gradual_unfreezing": true, # During first epoch the first group of layers is trained only. Starting second epoch, both groups are trained.
            "discriminative_fine_tuning": true, # Enable discriminative finetuning.
            "decay_factor": 0.01, # We want base model to be trained with learning rate 100 times smaller than heads.
            "cut_frac": 0.0, # Increase learning rate from the smallest to the base value instantly.
            "ratio": 32, # The ratio of the smallest to the largest (base) learning rate.
        },
        "callbacks": [
            { # Enable Tensorboard logs. Can we viewed via "tensorboard --logdir serialization_dir".
                "type": "tensorboard",
                "should_log_parameter_statistics": false,
                "should_log_learning_rate": true,
            }
        ],
        "num_epochs": 10,
        "validation_metric": "+Avg", # Track average score of all scores. '+' stands for 'higher - better'.
        "grad_clipping": 5.0, # Clip gradient if too high.
        "cuda_device": 0, # GPU
    }
}
