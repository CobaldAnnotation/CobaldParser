import sys
import os
import argparse
import shutil

import json

import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.dataset import CobaldJointDataset
from src.processing import NO_ARC_LABEL, postprocess
from src.vocabulary import Vocabulary
from src.parser import MorphoSyntaxSemanticsParser
from src.train import train_multiple_epochs
from src.predict import predict


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


def train_cmd(train_conllu_file, val_conllu_file, config_file, serialization_dir, device):
    # Read configuration json.
    with open(config_file, 'r') as file:
        config: dict = json.load(file)

    # Create raw training dataset to build vocabulary upon.
    raw_train_dataset = CobaldJointDataset(train_conllu_file)
    # Label namespaces.
    namespaces = [
        "lemma_rules",
        "joint_pos_feats",
        "deps_ud",
        "deps_eud",
        "miscs",
        "deepslots",
        "semclasses"
    ]
    # Build training vocabulary that maps string labels into integers.
    vocab = Vocabulary(raw_train_dataset, namespaces)
    # Make sure absent arcs have a value of -1, because positive values
    # indicate dependency relations ids.
    vocab.replace_index(NO_ARC_LABEL, -1, namespace="deps_ud")
    vocab.replace_index(NO_ARC_LABEL, -1, namespace="deps_eud")

    # Do not list number of classes in configuration file.
    # Instead, automatically fill them at runtime for convenience.
    tagger_args = config["model_args"]["tagger_args"]
    tagger_args["lemma_rule_classifier_args"]["n_classes"] = vocab.get_namespace_size("lemma_rules")
    tagger_args["pos_feats_classifier_args"]["n_classes"] = vocab.get_namespace_size("joint_pos_feats")
    tagger_args["depencency_classifier_args"]["n_rels_ud"] = vocab.get_namespace_size("deps_ud")
    tagger_args["depencency_classifier_args"]["n_rels_eud"] = vocab.get_namespace_size("deps_eud")
    tagger_args["misc_classifier_args"]["n_classes"] = vocab.get_namespace_size("miscs")
    tagger_args["deepslot_classifier_args"]["n_classes"] = vocab.get_namespace_size("deepslots")
    tagger_args["semclass_classifier_args"]["n_classes"] = vocab.get_namespace_size("semclasses")

    # Create actual training and validation datasets.
    transform = lambda sample: vocab.encode(sample)
    train_dataset = CobaldJointDataset(train_conllu_file, transform)
    val_dataset = CobaldJointDataset(val_conllu_file, transform)

    # Create dataloaders.
    g = torch.Generator()
    g.manual_seed(42)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        collate_fn=CobaldJointDataset.collate_fn,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        collate_fn=CobaldJointDataset.collate_fn,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    # Create model.
    model = MorphoSyntaxSemanticsParser(**config["model_args"])

    # Train model.
    optimizer = AdamW(model.parameters(), **config["optimizer_args"])
    best_model = train_multiple_epochs(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        config["n_epochs"],
        device
    )

    # Save the best model along with its vocabulary to a disk.
    vocab_path = os.path.join(serialization_dir, "vocab.json")
    vocab.serialize(vocab_path)
    model_path = os.path.join(serialization_dir, "model.bin")
    torch.save(model, model_path)
    # Also serialize input configuration, so that we can describe the model.
    shutil.copy(config, serialization_dir)

def predict_cmd(
    input_conllu_file,
    output_conllu_file,
    serialization_dir,
    batch_size,
    device
):
    # Create test dataloader.
    test_dataset = CobaldJointDataset(input_conllu_file)
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
    # Run model.
    predictions_int: list[dict[str, int]] = predict(model, test_dataloader, device)

    # Load training vocabulary.
    vocab_path = os.path.join(serialization_dir, "vocab.json")
    vocab = Vocabulary.deserialize(vocab_path)
    # Decode predictions from indexes to string labels.
    predictions_str: list[dict[str, str]] = [
        vocab.decode(prediction) for prediction in predictions_int
    ]
    # Post-process string labels (e.g. split joint morphological features
    # into upos, xpos and feats).
    predictions: list[dict[str, str]] = [
        postprocess(**prediction) for prediction in predictions_str
    ]
    with open(output_conllu_file, 'w') as file:
        for prediction in predictions:
            file.write(prediction.serialize())


def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(
        description="A simple application for model training and inference."
    )

    # Subparsers for mode-specific arguments
    subparsers = parser.add_subparsers(dest="subparser_name")

    # Train mode arguments
    train_parser = subparsers.add_parser("train", help="Arguments for training mode.")
    train_parser.add_argument(
        "train_conllu_file",
        type=str,
        help="Path to a training .conllu file."
    )
    train_parser.add_argument(
        "val_conllu_file",
        type=str,
        help="Path to a validation .conllu file."
    )
    train_parser.add_argument(
        "config_file",
        type=str,
        help="Path to a training configuration file."
    )
    train_parser.add_argument(
        "serialization_dir",
        type=str,
        help="Path to a model serialization directory. Must be empty."
    )

    # Predict mode arguments
    predict_parser = subparsers.add_parser("predict", help="Arguments for prediction mode.")
    predict_parser.add_argument(
        "input_conllu_file",
        type=str,
        help="Path to a conllu file to read unlabeled sentences from."
    )
    predict_parser.add_argument(
        "output_conllu_file",
        type=str,
        help="Path to a conllu file to write predictions to."
    )
    predict_parser.add_argument(
        "serialization_dir",
        type=str,
        help="Path to a serialization directory with saved model that will be used for inference."
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
            args.train_conllu_file,
            args.val_conllu_file,
            args.config_file,
            args.serialization_dir,
            device
        )
    elif args.subparser_name == "predict":
        predict_cmd(
            args.input_conllu_file,
            args.output_conllu_file,
            args.serialization_dir,
            args.batch_size,
            device
        )
    else:
        print("Invalid mode. Use 'train' or 'predict'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
