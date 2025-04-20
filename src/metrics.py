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


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    # Fields are ordered according to TrainingArguments.label_names.
    assert len(preds) == len(labels) == 8
    null_f1 = f1_score(preds[0].flatten(), labels[0].flatten(), average='macro')
    lemma_f1 = f1_score(preds[1].flatten(), labels[1].flatten(), average='macro')
    morph_f1 = f1_score(preds[2].flatten(), labels[2].flatten(), average='macro')
    ud_syntax_jaccard = jaccard_score_vectorwise(preds[3], labels[3])
    eud_syntax_jaccard = jaccard_score_vectorwise(preds[4], labels[4])
    miscs_f1 = f1_score(preds[5].flatten(), labels[5].flatten(), average='macro')
    deepslot_f1 = f1_score(preds[6].flatten(), labels[6].flatten(), average='macro')
    semclass_f1 = f1_score(preds[7].flatten(), labels[7].flatten(), average='macro')
    return {
        "null_f1": null_f1,
        "lemma_f1": lemma_f1,
        "morphology_f1": morph_f1,
        "ud_jaccard": ud_syntax_jaccard,
        "eud_jaccard": eud_syntax_jaccard,
        "miscs_f1": miscs_f1,
        "deepslot_f1": deepslot_f1,
        "semclass_f1": semclass_f1
    }
