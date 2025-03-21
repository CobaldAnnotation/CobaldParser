from typing import override
from copy import deepcopy

import numpy as np

import torch
from torch import nn
from torch import Tensor, BoolTensor, LongTensor
import torch.nn.functional as F

from src.activations import get_activation_fn
from src.bilinear_matrix_attention import BilinearMatrixAttention
from src.chu_liu_edmonds import decode_mst
from src.utils import pairwise_mask, replace_masked_values, IGNORE_INDEX


class DependencyHeadBase(nn.Module):
    """
    Base class for scoring arcs and relations between tokens in a dependency tree/graph.
    """

    def __init__(self, hidden_size: int, n_rels: int):
        super().__init__()

        self.arc_attention = BilinearMatrixAttention(
            hidden_size,
            hidden_size,
            use_input_biases=True,
            n_labels=1
        )
        self.rel_attention = BilinearMatrixAttention(
            hidden_size,
            hidden_size,
            use_input_biases=True,
            n_labels=n_rels
        )

    def forward(
        self,
        h_arc_head: Tensor,    # [batch_size, seq_len, hidden_size]
        h_arc_dep: Tensor,     # ...
        h_rel_head: Tensor,    # ...
        h_rel_dep: Tensor,     # ...
        gold_arcs: BoolTensor, # [batch_size, seq_len, seq_len]
        gold_rels: LongTensor, # [batch_size, seq_len, seq_len]
        mask: BoolTensor       # [batch_size, seq_len]
    ) -> dict[str, Tensor]:

        # Score arcs.
        # s_arc[:, i, j] = score of edge j -> i.
        s_arc = self.arc_attention(h_arc_head, h_arc_dep)
        # Mask undesirable values (padding, nulls, etc.) with -inf.
        replace_masked_values(s_arc, pairwise_mask(mask), replace_with=-1e8)
        # Score arc relations.
        # - [batch_size, seq_len, seq_len, num_labels]
        s_rel = self.rel_attention(h_rel_head, h_rel_dep).permute(0, 2, 3, 1)

        # Calculate loss.
        loss = torch.tensor(0.)
        if gold_arcs is not None and gold_rels is not None:
            arc_loss, rel_loss = self.calc_loss(s_arc, s_rel, gold_arcs, gold_rels, mask)
            # Aggregate both losses into one.
            loss = arc_loss + rel_loss

        # Predict arcs based on the scores.
        pred_arcs = self.predict_arcs(s_arc, mask)
        # Greedily select the most probable relation for each possible arc.
        # - [batch_size, seq_len, seq_len]
        pred_rels = s_rel.argmax(dim=-1)
        # Select relations towards predicted arcs.
        pred_rels = torch.where(pred_arcs.bool(), pred_rels, -torch.ones_like(pred_rels))

        return {
            # Return relations only, because arcs are inferred from it trivially.
            'preds': pred_rels,
            'loss': loss
        }

    def predict_arcs(
        self,
        s_arc: Tensor,   # [batch_size, seq_len, seq_len]
        mask: BoolTensor # [batch_size, seq_len]
    ) -> Tensor:
        """Predict arcs from scores."""
        raise NotImplementedError

    def calc_loss(
        self,
        s_arc: Tensor,         # [batch_size, seq_len, seq_len]
        s_rel: Tensor,         # [batch_size, seq_len, seq_len, num_labels]
        gold_arcs: BoolTensor, # [batch_size, seq_len, seq_len]
        gold_rels: LongTensor, # [batch_size, seq_len, seq_len]
        mask: BoolTensor       # [batch_size, seq_len]
    ) -> tuple[Tensor, Tensor]:
        """Calculate arc and relation loss."""
        raise NotImplementedError


class DependencyHead(DependencyHeadBase):
    """
    Basic UD syntax specialization that predicts single edge for each token.
    """

    @override
    def predict_arcs(
        self,
        s_arc: Tensor,   # [batch_size, seq_len, seq_len]
        mask: BoolTensor # [batch_size, seq_len]
    ) -> Tensor:

        if self.training:
            # During training, use fast greedy decoding.
            # - [batch_size, seq_len]
            pred_arcs_seq = s_arc.argmax(dim=-1)
        else:
            # During inference, diligently decode Maximum Spanning Tree.
            # FIXME
            # pred_arcs_seq = self._mst_decode(s_arc, mask)
            pred_arcs_seq = s_arc.argmax(dim=-1)

        # Upscale arcs sequence of shape [batch_size, seq_len]
        # to matrix of shape [batch_size, seq_len, seq_len].
        pred_arcs = F.one_hot(pred_arcs_seq, num_classes=pred_arcs_seq.size(1))
        # Apply mask.
        pred_arcs = pred_arcs * pairwise_mask(mask)
        return pred_arcs

    def _mst_decode(
        self,
        s_arc: Tensor, # [batch_size, seq_len, seq_len]
        mask: Tensor   # [batch_size, seq_len]
    ) -> tuple[Tensor, Tensor]:

        batch_size = s_arc.size(0)
        device = s_arc.device
        s_arc = s_arc.cpu()

        # Convert scores to probabilities, as `decode_mst` expects non-negative values.
        arc_probs = nn.functional.softmax(s_arc, dim=-1)
        # Transpose arcs, because decode_mst defines 'energy' matrix as
        #  energy[i,j] = "Score that `i` is the head of `j`",
        # whereas
        #  arc_probs[i,j] = "Probability that `j` is the head of `i`".
        arc_probs = arc_probs.transpose(1, 2)

        # `decode_mst` knows nothing about UD and ROOT, so we have to manually
        # zero probabilities of arcs leading to ROOT to make sure ROOT is a source node
        # of a graph.

        # Decode ROOT positions from diagonals.
        # shape: [batch_size]
        root_idxs = arc_probs.diagonal(dim1=1, dim2=2).argmax(dim=-1)
        # Zero out arcs leading to ROOTs.
        arc_probs[torch.arange(batch_size), :, root_idxs] = 0.0

        pred_arcs = []
        for sample_idx in range(batch_size):
            energy = arc_probs[sample_idx]
            # has_labels=False because we will decode them manually later.
            lengths = mask[sample_idx].sum()
            heads, _ = decode_mst(energy, lengths, has_labels=False)
            # Some nodes may be isolated. Pick heads greedily in this case.
            heads[heads <= 0] = s_arc[sample_idx].argmax(dim=-1)[heads <= 0]
            pred_arcs.append(heads)

        # shape: [batch_size, seq_len]
        pred_arcs = torch.from_numpy(np.stack(pred_arcs)).long().to(device)
        return pred_arcs

    @override
    def calc_loss(
        self,
        s_arc: Tensor,         # [batch_size, seq_len, seq_len]
        s_rel: Tensor,         # [batch_size, seq_len, seq_len, num_labels]
        gold_arcs: BoolTensor, # [batch_size, seq_len, seq_len]
        gold_rels: LongTensor, # [batch_size, seq_len, seq_len]
        mask: BoolTensor       # [batch_size, seq_len]
    ) -> tuple[Tensor, Tensor]:
        # Decompose gold matrix to gold heads and deprels.
        # tensor.max returns tuple (values, indices).
        # Values are gold relations, whereas indices are gold heads.
        # Both of shape [batch_size, seq_len].
        gold_deprels, gold_heads = gold_rels.max(dim=-1)
        # Calculate arc loss for all arcs (except for padded).
        arc_loss = F.cross_entropy(s_arc[mask], gold_heads[mask])
        # Calculate relation loss only at gold arcs.
        rel_loss = F.cross_entropy(s_rel[gold_arcs], gold_deprels[mask])
        return arc_loss, rel_loss


class MultiDependencyHead(DependencyHeadBase):
    """
    Enhanced UD syntax specialization that predicts multiple edges for each token.
    """

    @override
    def predict_arcs(
        self,
        s_arc: Tensor,   # [batch_size, seq_len, seq_len]
        mask: BoolTensor # [batch_size, seq_len]
    ) -> Tensor:

        # Convert scores to probabilities.
        arc_probs = torch.sigmoid(s_arc)
        # Find confident arcs (with prob > 0.5).
        pred_arcs = arc_probs.round().long()
        # Apply mask.
        pred_arcs = pred_arcs * pairwise_mask(mask)
        return pred_arcs

    @override
    def calc_loss(
        self,
        s_arc: Tensor,         # [batch_size, seq_len, seq_len]
        s_rel: Tensor,         # [batch_size, seq_len, seq_len, num_labels]
        gold_arcs: BoolTensor, # [batch_size, seq_len, seq_len]
        gold_rels: LongTensor, # [batch_size, seq_len, seq_len]
        mask: BoolTensor       # [batch_size, seq_len]
    ) -> tuple[Tensor, Tensor]:

        mask2d = pairwise_mask(mask)
        arc_loss = F.binary_cross_entropy_with_logits(s_arc[mask2d], gold_arcs[mask2d].float())
        rel_loss = F.cross_entropy(s_rel[gold_arcs], gold_rels[gold_arcs])
        return arc_loss, rel_loss


class DependencyClassifier(nn.Module):
    """
    Dozat and Manning's biaffine dependency classifier.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_rels_ud: int,
        n_rels_eud: int,
        activation: str,
        dropout: float,
    ):
        super().__init__()

        self.arc_dep_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            get_activation_fn(activation),
            nn.Dropout(dropout)
        )
        # All mlps are equal.
        self.arc_head_mlp = deepcopy(self.arc_dep_mlp)
        self.rel_dep_mlp = deepcopy(self.arc_dep_mlp)
        self.rel_head_mlp = deepcopy(self.arc_dep_mlp)

        self.dependency_head_ud = DependencyHead(hidden_size, n_rels_ud)
        self.dependency_head_eud = MultiDependencyHead(hidden_size, n_rels_eud)

    def forward(
        self,
        embeddings: Tensor, # [batch_size, seq_len, embedding_size]
        gold_ud: Tensor,    # [batch_size, seq_len, seq_len]
        gold_eud: Tensor,   # [batch_size, seq_len, seq_len]
        mask_ud: Tensor,    # [batch_size, seq_len]
        mask_eud: Tensor    # [batch_size, seq_len]
    ) -> dict[str, Tensor]:

        # - [batch_size, seq_len, hidden_size]
        h_arc_head = self.arc_head_mlp(embeddings)
        h_arc_dep = self.arc_dep_mlp(embeddings)
        h_rel_head = self.rel_head_mlp(embeddings)
        h_rel_dep = self.rel_dep_mlp(embeddings)

        # Share the h vectors between dependency and multi-dependency heads.
        output_ud = self.dependency_head_ud(
            h_arc_head,
            h_arc_dep,
            h_rel_head,
            h_rel_dep,
            gold_arcs=(gold_ud != IGNORE_INDEX), # Absent arcs have value of IGNORE_INDEX.
            gold_rels=gold_ud,
            mask=mask_ud
        )
        output_eud = self.dependency_head_eud(
            h_arc_head,
            h_arc_dep,
            h_rel_head,
            h_rel_dep,
            gold_arcs=(gold_eud != IGNORE_INDEX),
            gold_rels=gold_eud,
            mask=mask_eud
        )

        return {
            'preds_ud': output_ud["preds"],
            'preds_eud': output_eud["preds"],
            'loss_ud': output_ud["loss"],
            'loss_eud': output_eud["loss"]
        }
