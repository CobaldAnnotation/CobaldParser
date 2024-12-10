import ast

import torch
from torch import tensor


def recursive_find_unique(data) -> set:
    """
    Recursively find all unique elements in a list or nested lists.

    :param data: The list (or nested lists) to process.
    :return: A set of all unique elements found.
    """
    if isinstance(data, str) or isinstance(data, int) or isinstance(data, bool):
        return {data}

    unique_elements = set()
    for item in data:
        unique_elements |= recursive_find_unique(item)
    return unique_elements

def recursive_replace(data, transform):
    """
    Recursively replace elements in a list or nested lists according to a replacement map.

    :param data: The list (or nested lists) to process.
    :param replace_map: A dictionary mapping elements to their replacements.
    :return: A new list with elements replaced.
    """
    if isinstance(data, list):
        # Process each element in the list
        return [recursive_replace(element, transform) for element in data]
    else:
        # Replace the element if it's in the map, otherwise return as is
        return transform(data)


def build_condition_mask(sentences: list[list[str]], condition_fn: callable, device) -> tensor:
    masks = [
        torch.tensor([condition_fn(word) for word in sentence], dtype=bool, device=device)
        for sentence in sentences
    ]
    return torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False)

def build_padding_mask(sentences: list[list[str]], device) -> tensor:
    return build_condition_mask(sentences, condition_fn=lambda word: True, device=device)

def build_null_mask(sentences: list[list[str]], device) -> tensor:
    return build_condition_mask(sentences, condition_fn=lambda word: word == "#NULL", device=device)


def dict_from_str(s: str) -> dict:
    """Convert a string representation of a dict to a dict. (Yes, one cannot simply convert str to dict...)"""
    return ast.literal_eval(s)


def pad_matrices(matrices: list[tensor], padding_value: int = 0) -> tensor:
    """
    Pad square matrices so that each matrix is padded to the right and bottom.
    Basically a torch.nn.utils.rnn.pad_sequence for matrices.
    """
    # Determine the maximum size in each dimension
    max_height = max(t.size(0) for t in matrices)
    max_width = max(t.size(1) for t in matrices)
    assert max_height == max_width, "UD and E-UD matrices must be square."

    # Create a single tensor for all matrices
    padded_tensor = torch.full((len(matrices), max_height, max_width), padding_value)

    # Stack tensors directly into the larger tensor
    for i, matrix in enumerate(matrices):
        padded_tensor[i, :matrix.size(0), :matrix.size(1)] = matrix
    return padded_tensor


def pairwise_mask(masks1d: tensor) -> tensor:
    """
    Calculate an outer product of a mask, i.e. masks2d[:, i, j] = masks1d[:, i] * masks1d[:, j].
    Example:
    >>> masks1d = tensor([[True, True,  True, False],
                          [True, True, False, False]])
    >>> pairwise_mask(masks1d)
        tensor([[[ True,  True,  True, False],
                 [ True,  True,  True, False],
                 [ True,  True,  True, False],
                 [False, False, False, False]],

                [[ True,  True, False, False],
                 [ True,  True, False, False],
                 [False, False, False, False],
                 [False, False, False, False]]])
    """
    return masks1d[:, None, :] * masks1d[:, :, None]


# Credits: https://docs.allennlp.org/main/api/nn/util/#replace_masked_values
def replace_masked_values(tensor: tensor, mask: tensor, replace_with: float):
    assert tensor.dim() == mask.dim(), "tensor.dim() of {tensor.dim()} != mask.dim() of {mask.dim()}"
    tensor.masked_fill_(~mask, replace_with)


def align_two_sentences(lhs: list[str], rhs: list[str]) -> tuple:
    """
    Aligns two sequences of tokens. Empty token is inserted where needed.
    Example:
    >>> true_tokens = ["How", "did", "this", "#NULL", "happen"]
    >>> pred_tokens = ["How", "#NULL", "did", "this", "happen"]
    >>> align_labels(true_tokens, pred_tokens)
    ['How', '#EMPTY', 'did', 'this',  '#NULL', 'happen'],
    ['How',  '#NULL', 'did', 'this', '#EMPTY', 'happen']
    """
    lhs_aligned, rhs_aligned = [], []

    i, j = 0, 0
    while i < len(lhs) and j < len(rhs):
        if lhs[i] == "#NULL" and rhs[j] != "#NULL":
            lhs_aligned.append(lhs[i])
            rhs_aligned.append("#EMPTY")
            i += 1
        elif lhs[i] != "#NULL" and rhs[j] == "#NULL":
            lhs_aligned.append("#EMPTY")
            rhs_aligned.append(rhs[j])
            j += 1
        else:
            assert lhs[i] == rhs[j]
            lhs_aligned.append(lhs[i])
            rhs_aligned.append(rhs[j])
            i += 1
            j += 1

    if i < len(lhs):
        # lhs has extra #NULLs at the end, so append #EMPTY node to rhs
        assert j == len(rhs)
        while i < len(lhs):
            lhs_aligned.append(lhs[i])
            rhs_aligned.append("#NULL")
            i += 1
    if j < len(rhs):
        assert i == len(lhs)
        while j < len(rhs):
            lhs_aligned.append("#NULL")
            rhs_aligned.append(rhs[j])
            j += 1

    assert len(lhs_aligned) == len(rhs_aligned)
    return lhs_aligned, rhs_aligned

def align_sentences(lhs: list[list[str]], rhs: list[list[str]]) -> tuple:
    return zip(*[align_two_sentences(l, r) for l, r in zip(lhs, rhs)])

