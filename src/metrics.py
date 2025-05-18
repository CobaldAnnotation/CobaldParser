import numpy as np
from sklearn.metrics import f1_score
from transformers import EvalPrediction


def jaccard_score_vectorwise(pred_arcs: np.ndarray, gold_arcs: np.ndarray) -> float:
    """
    Compute Jaccard score between two sets of vectors.
    
    Args:
        pred_arcs: Predicted arcs as a 2D array with shape [n_pred_arcs, X]
        gold_arcs: Gold standard arcs as a 2D array with shape [n_gold_arcs, X]
        
    Returns:
        float: Jaccard similarity score between predicted and gold arcs
    """
    assert pred_arcs.ndim == gold_arcs.ndim == 2
    assert pred_arcs.shape[1] == gold_arcs.shape[1]

    # Convert tensors to sets of tuples for comparison
    pred_set = set(map(tuple, pred_arcs))
    gold_set = set(map(tuple, gold_arcs))

    # Calculate intersection and union
    intersection = pred_set.intersection(gold_set)
    union = pred_set.union(gold_set)
    if len(union) == 0:
        return 1.0 if len(intersection) == 0 else 0.0
    return len(intersection) / len(union)


# EvalPrediction have no information about preds/labels keys (why not just
# preserve dicts??). As a result, we cannot distinguish between tuples without
# external context, i.e. if `preds` is a tuple of length 3, it can contain
# (lemma rules, ud syntax, miscs) or (lemma rules, morph, semclass) - there is
# no way to find out which of these configurations we are dealing with and
# what scores we should calculate.
# So here is a workaround: pass extra `columns` that has the information about what
# columns are used, then bind this argument as
# `compute_metrics = lambda x: compute_metrics(x, columns)` at main.py when columns
# are known.
def compute_metrics(eval_pred: EvalPrediction, columns: list[str]) -> dict[str, float]:
    # preds and labels are aligned and ordered according
    # to TrainingArguments.label_names.
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    assert len(preds) == len(labels)

    result = {
        "null_f1": f1_score(preds[0].flatten(), labels[0].flatten(), average='macro')
    }

    current_position = 1
    if "lemma_rules" in columns:
        result["lemma_f1"] = f1_score(
            preds[current_position].flatten(),
            labels[current_position].flatten(),
            average='macro'
        )
        current_position += 1

    if "joint_feats" in columns:
        result["morphology_f1"] = f1_score(
            preds[current_position].flatten(),
            labels[current_position].flatten(),
            average='macro'
        )
        current_position += 1

    if "deps_ud" in columns:
        result["ud_jaccard"] = jaccard_score_vectorwise(
            preds[current_position],
            labels[current_position]
        )
        current_position += 1

    if "deps_eud" in columns:
        result["eud_jaccard"] = jaccard_score_vectorwise(
            preds[current_position],
            labels[current_position]
        )
        current_position += 1

    if "miscs" in columns:
        result["miscs_f1"] = f1_score(
            preds[current_position].flatten(),
            labels[current_position].flatten(),
            average='macro'
        )
        current_position += 1

    if "deepslots" in columns:
        result["deepslot_f1"] = f1_score(
            preds[current_position].flatten(),
            labels[current_position].flatten(),
            average='macro'
        )
        current_position += 1

    if "semclasses" in columns:
        result["semclass_f1"] = f1_score(
            preds[current_position].flatten(),
            labels[current_position].flatten(),
            average='macro'
        )
        current_position += 1

    return result