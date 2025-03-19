import torch
from torch import nn
from torch import Tensor

from src.mlp_classifier import MlpClassifier
from src.encoder import MaskedLanguageModelEncoder
from src.utils import build_padding_mask, pad_sequences


class NullPredictor(nn.Module):
    """
    A pipeline that restores ellipted tokens.
    """

    def __init__(
        self,
        encoder: MaskedLanguageModelEncoder,
        hidden_size: int,
        activation: str,
        dropout: float,
        consecutive_null_limit: int,
        class_weights: list[float] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.null_classifier = MlpClassifier(
            input_size=self.encoder.get_embedding_size(),
            hidden_size=hidden_size,
            n_classes=consecutive_null_limit + 1,
            activation=activation,
            dropout=dropout,
            class_weights=class_weights
        )

    def forward(self, words: list[list[str]], is_inference: bool) -> dict[str, any]:
        # Extra [CLS] token accounts for the case when #NULL is the first token in a sentence.
        words_with_cls = self._prepend_cls(words)

        # Delete nulls.
        words_without_nulls = self._del_nulls(words_with_cls)
        # Embeddings of words without nulls.
        embeddings_without_nulls = self.encoder(words_without_nulls)

        # Build padding mask.
        padding_mask_without_nulls = build_padding_mask(
            words_without_nulls,
            embeddings_without_nulls.device
        )

        # Build targets (if not at inference).
        if is_inference:
            gold_counting_mask = None
        else:
            gold_counting_mask = self._build_counting_mask(words_with_cls)
            gold_counting_mask = gold_counting_mask.to(embeddings_without_nulls.device)

        # Predict counting mask.
        classifier_output = self.null_classifier(
            embeddings_without_nulls,
            gold_counting_mask,
            padding_mask_without_nulls
        )

        # Add predicted nulls to the original sentences.
        pred_counting_mask = classifier_output["preds"]
        words_with_nulls = self._add_nulls(words, pred_counting_mask)
        return {
            "words": words_with_nulls,
            "loss": classifier_output["loss"]
        }

    @staticmethod
    def _build_counting_mask(sentences: list[list[str]]) -> Tensor:
        """
        Count the number of nulls following each non-null token for a bunch of sentences.
        output[i, j] = N means j-th non-null token in i-th sentence in followed by N nulls.
        """
        counting_masks: list[Tensor] = []

        for sentence in sentences:
            nonnull_words_idxs = [i for i, word in enumerate(sentence) if word != "#NULL"]
            nonnull_words_idxs.append(len(sentence))
            nonnull_words_idxs = torch.tensor(nonnull_words_idxs, dtype=torch.int)
            counting_mask = torch.diff(nonnull_words_idxs) - 1
            counting_masks.append(counting_mask)

        # Use -1 to make sure it is never attended to.
        # (if it is, the CUDA will terminate with an error that classes must be positive).
        counting_masks_batched = pad_sequences(counting_masks, padding_value=-1)
        return counting_masks_batched.long()

    @staticmethod
    def _prepend_cls(sentences: list[list[str]]) -> list[list[str]]:
        """
        Return a copy of sentences with [CLS] token prepended.
        """
        return [["[CLS]", *sentence] for sentence in sentences]

    @staticmethod
    def _del_nulls(sentences: list[list[str]]) -> list[list[str]]:
        """
        Return a copy of sentences with nulls removed.
        """
        return [[word for word in sentence if word != "#NULL"] for sentence in sentences]

    @staticmethod
    def _add_nulls(sentences: list[list[str]], counting_masks: Tensor) -> list[list[str]]:
        """
        Return a copy of sentences with nulls restored according to counting masks.
        """
        sentences_with_nulls = []
        for sentence, counting_mask in zip(sentences, counting_masks):
            sentence_with_nulls = []
            for word, n_nulls_to_insert in zip(sentence, counting_mask):
                sentence_with_nulls.append(word)
                for _ in range(1, n_nulls_to_insert + 1):
                    sentence_with_nulls.append("#NULL")
            sentences_with_nulls.append(sentence_with_nulls)
        return sentences_with_nulls
