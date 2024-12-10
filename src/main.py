import argparse
import sys

import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from dataset import CobaldJointDataset, NO_ARC_LABEL
from vocabulary import Vocabulary
from parser import MorphoSyntaxSemanticsParser
from dependency_classifier import NO_ARC_VALUE
from train import train_multiple_epochs
from predict import predict


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(mode=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def train_cmd(train_conllu_path, val_conllu_path, serialization_dir, batch_size, n_epochs, device):
    # Create raw training dataset to build vocabulary upon.
    raw_train_dataset = CobaldJointDataset(train_conllu_path)
    # Build training vocabulary that maps string labels into integers.
    vocab = Vocabulary(
        raw_train_dataset,
        # Namespaces to encode.
        namespaces=[
            "lemma_rules",
            "joint_pos_feats",
            "deps_ud",
            "deps_eud",
            "miscs",
            "deepslots",
            "semclasses"
        ]
    )
    # Make sure absent arcs have a value of -1, because positive values
    # indicate dependency relations.
    vocab.replace_index(NO_ARC_LABEL, NO_ARC_VALUE, namespace="deps_ud")
    vocab.replace_index(NO_ARC_LABEL, NO_ARC_VALUE, namespace="deps_eud")

    # Create actual training and validation datasets.
    transform = lambda sample: vocab.encode(sample)
    train_dataset = CobaldJointDataset(train_conllu_path, transform)
    val_dataset = CobaldJointDataset(val_conllu_path, transform)

    # Create dataloaders.
    g = torch.Generator()
    g.manual_seed(42)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=CobaldJointDataset.collate_fn,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=CobaldJointDataset.collate_fn,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    # Create model.
    model_args = {
        "encoder_args": {
            "model_name": "distilbert-base-uncased",
            "train_parameters": True
        },
        "null_predictor_args": {
            "hidden_size": 512,
            "activation": "relu",
            "dropout": 0.1,
            "consecutive_null_limit": 2
        },
        "tagger_args": {
            "lemma_rule_classifier_args": {
                "hidden_size": 512,
                "n_classes": vocab.get_namespace_size("lemma_rules"),
                "activation": "relu",
                "dropout": 0.1,
            },
            "pos_feats_classifier_args": {
                "hidden_size": 512,
                "n_classes": vocab.get_namespace_size("joint_pos_feats"),
                "activation": "relu",
                "dropout": 0.1,
            },
            "depencency_classifier_args": {
                "hidden_size": 128,
                "n_rels_ud": vocab.get_namespace_size("deps_ud"),
                "n_rels_eud": vocab.get_namespace_size("deps_eud"),
                "activation": "relu",
                "dropout": 0.1,
            },
            "misc_classifier_args": {
                "hidden_size": 256,
                "n_classes": vocab.get_namespace_size("miscs"),
                "activation": "relu",
                "dropout": 0.1,
            },
            "deepslot_classifier_args": {
                "hidden_size": 512,
                "n_classes": vocab.get_namespace_size("deepslots"),
                "activation": "relu",
                "dropout": 0.1,
            },
            "semclass_classifier_args": {
                "hidden_size": 512,
                "n_classes": vocab.get_namespace_size("semclasses"),
                "activation": "relu",
                "dropout": 0.1,
            }
        }
    }
    model = MorphoSyntaxSemanticsParser(**model_args)

    # Train model.
    optimizer = AdamW(model.parameters(), lr=3e-4)
    best_model = train_multiple_epochs(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        n_epochs,
        device
    )

    # Save the best model along with its vocabulary to a disk.
    vocab_path = os.path.join(serialization_dir, "vocab.json")
    vocab.serialize(vocab_path)
    model_path = os.path.join(serialization_dir, "model.bin")
    torch.save(model, model_path)


def predict_cmd(conllu_path, serialization_dir, batch_size, device):
    # Load training vocabulary.
    vocab_path = os.path.join(serialization_dir, "vocab.json")
    vocab = Vocabulary.deserialize(vocab_path)
    test_dataset = CobaldJointDataset(conllu_path)
    # Create test dataloader.
    g = torch.Generator()
    g.manual_seed(42)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=CobaldJointDataset.collate_fn,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    # Load model from a disk.
    model_path = os.path.join(serialization_dir, "model.bin")
    model = torch.load(model_path, weights_only=False)
    predictions = predict(model, test_dataloader, device)

    print(f"N predictions: {len(predictions)}\nexample: {predictions[0]}")


def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(
        description="A simple application for model training and prediction."
    )

    # Subparsers for mode-specific arguments
    subparsers = parser.add_subparsers(dest="subparser_name")

    # Train mode arguments
    train_parser = subparsers.add_parser("train", help="Arguments for training mode.")
    train_parser.add_argument(
        "train_conllu_path",
        type=str,
        help="Path to the training .conllu file."
    )
    train_parser.add_argument(
        "val_conllu_path",
        type=str,
        help="Path to the validation .conllu file."
    )
    train_parser.add_argument(
        "serialization_dir",
        type=str,
        help="Serialization path."
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for dataloaders."
    )
    train_parser.add_argument(
        "--n_epochs",
        type=int,
        default=1,
        help="Number of training epochs."
    )

    # Predict mode arguments
    predict_parser = subparsers.add_parser("predict", help="Arguments for prediction mode.")
    predict_parser.add_argument(
        "conllu_path",
        type=str,
        help="Path to the input .conllu file for prediction."
    )
    predict_parser.add_argument(
        "model_path",
        type=str,
        help="Path to a model that will be used for inference."
    )
    predict_parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for test dataloader."
    )

    args = parser.parse_args()

    if args.subparser_name == "train":
        # Create serialization directory and make sure it does not exist.
        os.makedirs(args.serialization_dir, exist_ok=False)
        train_cmd(
            args.train_conllu_path,
            args.val_conllu_path,
            args.serialization_dir,
            args.batch_size,
            args.n_epochs,
            device
        )
    elif args.subparser_name == "predict":
        predict_cmd(
            args.conllu_path,
            args.model_path,
            args.batch_size,
            device
        )
    else:
        print("Invalid mode. Use 'train' or 'predict'.")
        sys.exit(1)


if __name__ == "__main__":
    main()

