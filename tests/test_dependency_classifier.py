import pytest
import torch

from cobald_parser.dependency_classifier import DependencyHead, MultiDependencyHead


@pytest.fixture
def s_arc():
    # A constant for readability
    mask = -1e8
    # Create sample scores
    return torch.tensor(
        # batch 0, token 3 is padded
        [[[-1.42,  0.52,  0.17,  mask],
          [ 0.37,  0.19, -0.13,  mask],
          [ 1.66,  0.54,  1.03,  mask],
          [ mask,  mask,  mask,  mask]],
        # batch 1, token 2 is masked
         [[-0.56, -2.21,  mask, -0.54],
          [-1.37, -0.58,  mask, -1.06],
          [ mask,  mask,  mask,  mask],
          [-1.02, -2.50,  mask, -1.05]]]
    )

@pytest.fixture
def s_rel():
    return torch.tensor(
        [[[[-0.52, -1.04, -0.88,  0.05,  0.51],
           [-0.81, -0.21, -0.52,  1.18, -0.79],
           [ 0.01, -0.44, -0.41, -0.92, -1.45],
           [ 0.78,  0.10, -1.48, -0.51, -0.36]],
          [[-0.19, -0.27, -0.29, -0.00, -0.12],
           [-0.52, -0.55, -2.06, -1.29,  2.33],
           [-2.00, -0.25, -0.51,  1.33,  0.08],
           [ 0.86,  0.35, -0.12,  0.60,  0.81]],
          [[-0.75, -0.83, -0.52,  0.94, -0.06],
           [ 1.81,  0.45, -0.51,  0.56, -0.55],
           [-1.04,  0.21, -0.56, -1.02,  0.31],
           [-3.00, -0.01,  0.14, -0.77,  0.59]],
          [[-0.30, -1.20,  0.22,  0.09, -1.23],
           [-0.15,  0.39, -0.67, -1.53, -0.82],
           [-0.09, -0.19,  0.12,  0.62,  0.24],
           [-0.63, -0.46,  1.84,  1.14, -1.80]]],
         [[[-0.51,  0.01,  0.64, -0.86, -0.61],
           [-2.03, -0.47, -0.42, -0.22, -1.65],
           [-0.75, -0.20, -1.07, -0.15, -0.19],
           [-0.74, -1.21,  0.94, -0.55,  0.75]],
          [[ 0.97, -2.00,  0.13,  0.53,  0.60],
           [-0.63,  1.58, -1.33,  0.38, -0.15],
           [-2.05,  1.05, -0.62,  1.17, -0.20],
           [ 0.91, -0.28,  0.85,  0.25, -0.70]],
          [[-2.80, -0.62, -0.26,  1.44, -1.81],
           [-0.15, -0.83,  0.05, -0.57,  0.89],
           [ 0.56, -0.46, -0.52,  2.09, -0.84],
           [-0.04, -0.12,  0.22, -0.23,  1.74]],
          [[ 0.39, -1.79, -0.66, -1.27,  0.13],
           [-0.02,  2.41, -0.10,  0.87, -1.01],
           [-2.08,  0.65,  0.51, -0.69, -1.05],
           [-0.91, -0.40, -0.25, -0.54, -0.22]]]]
    )

@pytest.fixture
def gold_arcs():
    # Create gold arcs [n_arcs, 4] where the columns are [batch_idx, dep_idx, head_idx, rel_idx]
    return torch.tensor(
        [[0, 0, 1, 2],  # batch 0, token 0 has head at 1 with relation 2
         [0, 1, 2, 4],  # batch 0, token 1 has head at 2 with relation 4
         [0, 2, 0, 4],  # batch 0, token 2 has head at 0 with relation 4
                        # batch 0, token 3 is padded
         [1, 0, 1, 3],  # batch 1, token 0 has head at 1 with relation 3
         [1, 1, 1, 2],  # batch 1, token 1 has self-loop with relation 2
                        # batch 1, token 2 is masked
         [1, 3, 1, 0]], # batch 1, token 3 has head at 1 with relation 0
    )


class TestDependencyHead:

    @pytest.fixture(autouse=True)
    def setup(self, s_rel):
        self.dependency_head = DependencyHead(hidden_size=16, n_rels=s_rel.size(-1))

    def test_calc_arc_loss(self, s_arc, gold_arcs):
        arc_loss = self.dependency_head.calc_arc_loss(s_arc, gold_arcs)
        assert torch.isclose(arc_loss, torch.tensor(1.3446), atol=1e-4)

    def test_calc_rel_loss(self, s_rel, gold_arcs):
        rel_loss = self.dependency_head.calc_rel_loss(s_rel, gold_arcs)
        assert torch.isclose(rel_loss, torch.tensor(2.1603), atol=1e-4)

    # TODO: test predictions


class TestMultiDependencyHead:

    def test_calc_arc_loss(self, s_arc, gold_arcs):
        arc_loss = MultiDependencyHead.calc_arc_loss(s_arc, gold_arcs)
        assert torch.isclose(arc_loss, torch.tensor(0.4494), atol=1e-4)
