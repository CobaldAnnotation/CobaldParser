import torch
from torch import nn
from torch import Tensor, BoolTensor, LongTensor

from src.activations import get_activation_fn


class MlpClassifier(nn.Module):
    """ Simple feed-forward multilayer perceptron classifier. """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_classes: int,
        activation: str,
        dropout: float,
        class_weights: list[float] = None,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )
        class_weights_pt = torch.tensor(class_weights) if class_weights is not None else None
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights_pt)

    def forward(
        self,
        embeddings: Tensor,
        labels: LongTensor = None,
        mask: BoolTensor = None
    ) -> dict[str, Tensor]:

        logits = self.classifier(embeddings)

        # Calculate loss.
        loss = torch.tensor(0.)
        if labels is not None:
            loss = self.loss_fn(logits[mask], labels[mask])

        # Predictions.
        preds = logits.argmax(dim=-1)

        return {'preds': preds, 'loss': loss}
