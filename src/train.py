# Based on https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

from tqdm import tqdm
import numpy

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from metrics import CategoricalAccuracy, F1Measure, MultilabelAttachmentScore
from utils import build_padding_mask, build_null_mask, pairwise_mask, align_sentences


def evaluate(model: nn.Module, dataloader: DataLoader, device) -> dict[str, float]:
    model.eval()

    # Create scorers.
    # Lemma scorers.
    lemma_accuracy_scorer = CategoricalAccuracy()
    lemma_f1_scorer = F1Measure(average='macro')
    # POS & feats scorers.
    pos_feats_accuracy_scorer = CategoricalAccuracy()
    pos_feats_f1_scorer = F1Measure(average='macro')
    # Syntax scorers.
    ud_syntax_scorer = MultilabelAttachmentScore()
    eud_syntax_scorer = MultilabelAttachmentScore()
    # Deepslot scorers.
    deepslot_accuracy_scorer = CategoricalAccuracy()
    deepslot_f1_scorer = F1Measure(average='macro')
    # Semclass scorers.
    semclass_accuracy_scorer = CategoricalAccuracy()
    semclass_f1_scorer = F1Measure(average='macro')
    # Null scorer.
    # Null prediction is a binary classification, so micro F1 = binary F1.
    null_scorer = F1Measure(average='micro')

    sum_loss = 0.
    # Disable gradient computation.
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate"):
            for key, value in batch.items():
                batch[key] = value.to(device) if isinstance(value, torch.Tensor) else value
            output = model(**batch)
            sum_loss += output["loss"].item()

            # Build masks.
            padding_mask = build_padding_mask(batch["words"], device)
            null_mask = build_null_mask(batch["words"], device)
            # Update cumulative scores.
            lemma_accuracy_scorer.add(output["lemma_rules"], batch["lemma_rules"], padding_mask)
            lemma_f1_scorer.add(output["lemma_rules"], batch["lemma_rules"], padding_mask)
            # POS & feats.
            pos_feats_accuracy_scorer.add(
                output["joint_pos_feats"],
                batch["joint_pos_feats"],
                padding_mask
            )
            pos_feats_f1_scorer.add(
                output["joint_pos_feats"],
                batch["joint_pos_feats"],
                padding_mask
            )
            # Syntax.
            ud_syntax_scorer.add(
                output["deps_ud"],
                batch["deps_ud"],
                mask=pairwise_mask(padding_mask & ~null_mask)
            )
            eud_syntax_scorer.add(
                output["deps_eud"],
                batch["deps_eud"],
                mask=pairwise_mask(padding_mask)
            )
            # Score deepslots.
            deepslot_accuracy_scorer.add(output["deepslots"], batch["deepslots"], padding_mask)
            deepslot_f1_scorer.add(output["deepslots"], batch["deepslots"], padding_mask)
            # Score semclasses.
            semclass_accuracy_scorer.add(output["semclasses"], batch["semclasses"], padding_mask)
            semclass_f1_scorer.add(output["semclasses"], batch["semclasses"], padding_mask)
            # Score nulls.
            pred_words_aligned, gold_words_aligned = align_sentences(output["words"], batch["words"])
            pred_nulls_mask = build_null_mask(pred_words_aligned, device)
            gold_nulls_mask = build_null_mask(gold_words_aligned, device)
            padding_mask_aligned = build_padding_mask(gold_words_aligned, device)
            null_scorer.add(pred_nulls_mask, gold_nulls_mask, padding_mask_aligned)

    # Get the averaged scores.
    lemma_accuracy = lemma_accuracy_scorer.get_average()
    lemma_f1 = lemma_f1_scorer.get_average()["f1"]
    pos_feats_accuracy = pos_feats_accuracy_scorer.get_average()
    pos_feats_f1 = pos_feats_f1_scorer.get_average()["f1"]
    ud_syntax = ud_syntax_scorer.get_average()
    uas, las = ud_syntax["UAS"], ud_syntax["LAS"]
    eud_syntax = eud_syntax_scorer.get_average()
    euas, elas = eud_syntax["UAS"], eud_syntax["LAS"]
    deepslot_accuracy = deepslot_accuracy_scorer.get_average()
    deepslot_f1 = deepslot_f1_scorer.get_average()["f1"]
    semclass_accuracy = semclass_accuracy_scorer.get_average()
    semclass_f1 = semclass_f1_scorer.get_average()["f1"]
    null_f1 = null_scorer.get_average()["f1"]
    # Average averaged scores.
    average_f1 = numpy.mean([
        lemma_f1,
        pos_feats_f1,
        uas, las, euas, elas,
        deepslot_f1,
        semclass_f1,
        null_f1
    ])
    # Average loss.
    average_loss = sum_loss / len(dataloader)

    return {
        "Lemma accuracy": lemma_accuracy,
        "Lemma F1": lemma_f1,
        "POS-&-Feats accuracy": pos_feats_accuracy,
        "POS-&-Feats F1": pos_feats_f1,
        "UAS": uas,
        "LAS": las,
        "EUAS": euas,
        "ELAS": elas,
        "Deepslot accuracy": deepslot_accuracy,
        "Deepslot F1": deepslot_f1,
        "Semclass accuracy": semclass_accuracy,
        "Semclass F1": semclass_f1,
        "Null F1": null_f1,
        "Average F1": average_f1,
        "Loss": average_loss
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    epoch_index: int,
    device
) -> float:
    # Make sure gradient tracking is on and perform training loop.
    model.train()

    sum_loss = 0.
    # Use enumerate(), so that we can track batch index and do some intra-epoch reporting
    for batch in tqdm(dataloader, desc="Train"):
        optimizer.zero_grad()
        for key, value in batch.items():
            batch[key] = value.to(device) if isinstance(value, Tensor) else value
        output = model(**batch)
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    average_loss = sum_loss / len(dataloader)
    return average_loss


def train_multiple_epochs(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    n_epochs: int,
    device
) -> nn.Module:
    model.to(device)

    best_val_loss = float('inf')
    best_model = model
    for epoch_index in range(n_epochs):
        epoch_number = epoch_index + 1

        train_loss = train_one_epoch(model, train_dataloader, optimizer, epoch_index, device)

        # Evaluate model on validation data.
        val_metrics = evaluate(model, val_dataloader, device)
        val_loss = val_metrics.pop("Loss")

        print(f"======= Epoch {epoch_number} =======")
        print(f'Loss train: {train_loss:.4f}, val.: {val_loss:.4f}')
        for name, value in val_metrics.items():
            print(f"Val. {name}: {value:.4f}")
        print()

        # Update best model according to loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

    return best_model
