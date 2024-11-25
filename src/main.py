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
from train import train as train_model


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


def train(train_conllu_path, val_conllu_path, batch_size, n_epochs):
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
            "consecutive_null_limit": 4
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

    optimizer = AdamW(model.parameters(), lr=3e-4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, train_dataloader, val_dataloader, optimizer, n_epochs, device)


def predict(conllu_path):
    raise NotImplementedError


def main():
    seed_everything(42)

    parser = argparse.ArgumentParser(description="A simple application with train and predict modes.")

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

    args = parser.parse_args()

    if args.subparser_name == "train":
        train(args.train_conllu_path, args.val_conllu_path, args.batch_size, args.n_epochs)
    elif args.subparser_name == "predict":
        predict(args.conllu_path)
    else:
        print("Invalid mode. Use 'train' or 'predict'.")
        sys.exit(1)


if __name__ == "__main__":
    main()

