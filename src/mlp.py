from overrides import override

import torch
from torch import nn
from torch import Tensor, BoolTensor, LongTensor

import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy


ACT2FN = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh()
}


class MLP(nn.Model):
    """
    Multilayer perceptron.
    """
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        n_classes: int,
        activation: str,
        dropout: float
    ):
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hid_dim),
            ACT2FN[activation],
            nn.Dropout(dropout),
            nn.Linear(hid_dim, n_classes)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = CategoricalAccuracy()

    @override(check_signature=False)
    def forward(
        self,
        embeddings: Tensor,
        labels: LongTensor = None,
        mask: BoolTensor = None
    ) -> dict[str, Tensor]:

        logits = self.classifier(embeddings)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(-1)

        loss = torch.tensor(0.)
        if labels is not None:
            loss = self.loss(logits, labels, mask)
            self.update_metrics(logits, labels, mask)

        return {'preds': preds, 'probs': probs, 'loss': loss}

    def loss(self, logits: Tensor, labels: LongTensor, mask: BoolTensor) -> Tensor:
        return self.criterion(logits[mask], labels[mask])

    def update_metrics(self, logits: Tensor, labels: LongTensor, mask: BoolTensor):
        self.accuracy(logits, labels, mask)

    def get_metrics(self, reset: bool = False) -> dict[str, float]:
        return {"Accuracy": self.accuracy.get_metric(reset)}

