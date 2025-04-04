import pytest
import torch

from src.utils import (
    pad_sequences,
    build_padding_mask,
    build_null_mask,
    pairwise_mask,
    replace_masked_values
)


class TestPadSequences:
    def test_pad_sequences_equal_length(self):
        sequences = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6]),
        ]
        result = pad_sequences(sequences, padding_value=0)
        expected = torch.tensor([[1, 2, 3], [4, 5, 6]])
        assert torch.equal(result, expected)

    def test_pad_sequences_different_lengths(self):
        sequences = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
            torch.tensor([6]),
        ]
        result = pad_sequences(sequences, padding_value=0)
        expected = torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]])
        assert torch.equal(result, expected)


class TestBuildPaddingMask:
    def test_build_padding_mask(self):
        sentences = [["hello", "world"], ["test"]]
        result = build_padding_mask(sentences, device="cpu")
        expected = torch.tensor([[True, True], [True, False]])
        assert torch.equal(result, expected)

    def test_build_padding_mask_empty_sentences(self):
        sentences = [[], []]
        result = build_padding_mask(sentences, device="cpu")
        expected = torch.tensor([[], []], dtype=torch.bool)
        assert result.shape == (2, 0)


class TestBuildNullMask:
    def test_build_null_mask(self):
        sentences = [["hello", "#NULL"], ["#NULL", "test"]]
        result = build_null_mask(sentences, device="cpu")
        expected = torch.tensor([[False, True], [True, False]])
        assert torch.equal(result, expected)

    def test_build_null_mask_no_nulls(self):
        sentences = [["hello", "world"], ["test"]]
        result = build_null_mask(sentences, device="cpu")
        expected = torch.tensor([[False, False], [False, False]])
        assert torch.equal(result, expected)


class TestPairwiseMask:
    def test_pairwise_mask(self):
        masks1d = torch.tensor(
            [[  True,  True,  True, False],
             [  True, False,  True,  True]]
        )
        result = pairwise_mask(masks1d)
        expected = torch.tensor(
            [[[ True,  True,  True, False],
              [ True,  True,  True, False],
              [ True,  True,  True, False],
              [False, False, False, False]],
  
             [[ True, False,  True,  True],
              [False, False, False, False],
              [ True, False,  True,  True],
              [ True, False,  True,  True]]]
        )
        assert torch.equal(result, expected)


class TestReplaceMaskedValues:
    def test_replace_masked_values(self):
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[True, False, True], [True, True, False]])
        
        replace_masked_values(tensor, mask, replace_with=-1.0)
        
        expected = torch.tensor([[1.0, -1.0, 3.0], [4.0, 5.0, -1.0]])
        assert torch.equal(tensor, expected)

    def test_replace_masked_values_all_masked(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[False, False], [False, False]])

        replace_masked_values(tensor, mask, replace_with=0.0)
        
        expected = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        assert torch.equal(tensor, expected)

    def test_replace_masked_values_dimension_mismatch(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([True, False])
        
        with pytest.raises(AssertionError):
            replace_masked_values(tensor, mask, replace_with=0.0)