from overrides import override

import torch
from torch import Tensor

from utils import replace_masked_values


class Metric:
    """ A general abstract class representing a metric which can be accumulated. """

    def add(self, preds: Tensor, golds: Tensor, mask: Tensor = None):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions.
        gold_labels : `torch.Tensor`, required.
            A tensor corresponding to some gold label to evaluate against.
        mask : `torch.Tensor`, optional (default = `None`).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        """
        raise NotImplementedError

    def get_average(self):
        """
        Compute and return the metric.
        """
        raise NotImplementedError


class CategoricalAccuracy(Metric):
    def __init__(self):
        self.correct_count = 0
        self.total_count = 0

    @override
    def add(self, preds: Tensor, golds: Tensor, mask: Tensor = None):
        if mask is not None:
            preds = preds[mask]
            golds = golds[mask]
        # Ensure predictions and gold_labels are of the same shape
        assert preds.shape == golds.shape, "Shape mismatch between predictions and gold labels"
        self.correct_count += torch.sum(preds == golds)
        self.total_count += torch.numel(golds)

    @override
    def get_average(self):
        return self.correct_count / self.total_count


class F1Measure(Metric):
    def __init__(self, average: str = 'macro'):
        assert average in ['macro', 'micro', 'weighted']
        self.tp_sum = dict()
        self.fp_sum = dict()
        self.fn_sum = dict()
        self.average = average

    @override
    def add(self, preds: Tensor, golds: Tensor, mask: Tensor = None):
        if mask is not None:
            preds = preds[mask]
            golds = golds[mask]
        assert preds.shape == golds.shape, "Shape mismatch between predictions and gold labels"
        # Calculate true positives, false positives, and false negatives
        for label in torch.unique(torch.cat([preds, golds])):
            label = label.item()
            self.tp_sum[label] = self.tp_sum.get(label, 0) + ((preds == label) & (golds == label)).sum().item()
            self.fp_sum[label] = self.fp_sum.get(label, 0) + ((preds == label) & (golds != label)).sum().item()
            self.fn_sum[label] = self.fn_sum.get(label, 0) + ((preds != label) & (golds == label)).sum().item()

    @override
    def get_average(self) -> dict[str, float]:
        precisions = {}
        recalls = {}
        f1s = {}
        supports = {}

        for label in self.tp_sum.keys():
            tp = self.tp_sum.get(label, 0)
            fp = self.fp_sum.get(label, 0)
            fn = self.fn_sum.get(label, 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            precisions[label] = precision
            recalls[label] = recall
            f1s[label] = f1
            supports[label] = tp + fn

        if self.average == 'micro':
            total_tp = sum(self.tp_sum.values())
            total_fp = sum(self.fp_sum.values())
            total_fn = sum(self.fn_sum.values())

            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            return {"precision": precision, "recall": recall, "f1": f1}

        elif self.average == 'macro':
            precision = sum(precisions.values()) / len(precisions) if precisions else 0.0
            recall = sum(recalls.values()) / len(recalls) if recalls else 0.0
            f1 = sum(f1s.values()) / len(f1s) if f1s else 0.0

            return {"precision": precision, "recall": recall, "f1": f1}

        elif self.average == 'weighted':
            total_support = sum(supports.values())
            precision = sum(precisions[label] * supports[label] for label in supports) / total_support if total_support > 0 else 0.0
            recall = sum(recalls[label] * supports[label] for label in supports) / total_support if total_support > 0 else 0.0
            f1 = sum(f1s[label] * supports[label] for label in supports) / total_support if total_support > 0 else 0.0

            return {"precision": precision, "recall": recall, "f1": f1}

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


class MultilabelAttachmentScore(Metric):
    def __init__(self):
        self.unlabeled_score_sum = 0.0
        self.labeled_score_sum = 0.0
        self.total_count = 0.0

    @override
    def add(
        self,
        preds: Tensor, # [batch_size, seq_len, seq_len]
        golds: Tensor, # [batch_size, seq_len, seq_len]
        mask: Tensor = None
    ):
        assert preds.dim() == 3
        assert preds.shape == golds.shape, "Shape mismatch between predictions and gold labels"
        if mask is not None:
            assert preds.shape == mask.shape
            # Replace masked arcs with -1.
            replace_masked_values(preds, mask, -1)
            replace_masked_values(golds, mask, -1)

        # Select present arcs.
        # [batch_size, seq_len, seq_len]
        pred_arcs = (preds != -1)
        gold_arcs = (golds != -1)
        # [batch_size, seq_len, seq_len]
        intersecting_arcs = pred_arcs.logical_and(gold_arcs)
        # Sum along all dimentions but batch_size, because attachment scores must be
        # computed independently for each sample in a batch.
        # [batch_size]
        unlabeled_intersections_sizes = intersecting_arcs.sum(dim=[1, 2])
        labeled_intersections_sizes = (intersecting_arcs * (preds == golds)).sum(dim=[1, 2])
        # Element-wise maximum of non-zero arcs counts.
        # [batch_size]
        max_arcs_sizes = torch.maximum(
            torch.count_nonzero(pred_arcs, dim=[1, 2]),
            torch.count_nonzero(gold_arcs, dim=[1, 2])
        )

        # Calculate Intersection-over-Maximum (=attachment score).
        # Add 1e-8 to avoid division by zero.
        unlabeled_scores = unlabeled_intersections_sizes / (max_arcs_sizes + 1e-8)
        labeled_scores = labeled_intersections_sizes / (max_arcs_sizes + 1e-8)
        # Labeled attachment is always stricter than unlabeled attachment.
        assert torch.all(labeled_scores <= unlabeled_scores)

        # Sum attachment scores of batch's samples.
        self.unlabeled_score_sum += torch.sum(unlabeled_scores).item()
        self.labeled_score_sum += torch.sum(unlabeled_scores).item()
        batch_size = len(golds)
        self.total_count += batch_size

    @override
    def get_average(self):
        return {
            "UAS": self.unlabeled_score_sum / self.total_count,
            "LAS": self.labeled_score_sum / self.total_count
        }
