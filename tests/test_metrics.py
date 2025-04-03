import numpy as np
from src.metrics import jaccard_score_vectorwise


def test_jaccard_score_vectorwise_identical():
    # Test with identical tensors
    pred_arcs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    gold_arcs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    score = jaccard_score_vectorwise(pred_arcs, gold_arcs)
    assert score == 1.0

def test_jaccard_score_vectorwise_different():
    # Test with completely different tensors
    pred_arcs = np.array([[1, 2, 3], [4, 5, 6]])
    gold_arcs = np.array([[7, 8, 9], [10, 11, 12]])
    score = jaccard_score_vectorwise(pred_arcs, gold_arcs)
    assert score == 0.0

def test_jaccard_score_vectorwise_partial_overlap1():
    # Test with partial overlap
    pred_arcs = np.array([[1, 2, 3], [4, 5, 6]])
    gold_arcs = np.array([[1, 2, 3], [7, 8, 9]])
    score = jaccard_score_vectorwise(pred_arcs, gold_arcs)
    assert score == 1/3

def test_jaccard_score_vectorwise_partial_overlap2():
    # Test with partial overlap
    pred_arcs = np.array([[1, 2, 3], [4, 5, 6], [2, 3, 4]])
    gold_arcs = np.array([[1, 2, 3], [4, 5, 0], [2, 3, 4]])
    score = jaccard_score_vectorwise(pred_arcs, gold_arcs)
    assert score == 0.5

def test_jaccard_score_vectorwise_excess():
    # Test when more than necessary arcs were predicted 
    pred_arcs = np.array([[1, 2, 3], [4, 5, 6], [2, 3, 4]])
    gold_arcs = np.array([[1, 2, 3], [4, 5, 0]])
    score = jaccard_score_vectorwise(pred_arcs, gold_arcs)
    assert score == 0.25

def test_jaccard_score_vectorwise_deficit():
    # Test when not enough arcs were predicted 
    pred_arcs = np.array([[1, 2, 3], [4, 5, 0]])
    gold_arcs = np.array([[1, 2, 3], [4, 5, 6], [2, 3, 4]])
    score = jaccard_score_vectorwise(pred_arcs, gold_arcs)
    assert score == 0.25
