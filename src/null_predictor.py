from torch import nn
from torch import Tensor

from src.mlp_classifier import MlpClassifier
from src.encoder import MaskedLanguageModelEncoder


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

    def forward(
        self,
        words: list[list[str]],
        counting_mask: Tensor = None
    ) -> dict[str, any]:
        
        # Extra [CLS] token accounts for the case when #NULL is the first token in a sentence.
        words_with_cls = self._prepend_cls(words)
        words_without_nulls = self._remove_nulls(words_with_cls)
        # Embeddings of words without nulls.
        embeddings_without_nulls = self.encoder(words_without_nulls)

        # Predict counting mask.
        classifier_output = self.null_classifier(embeddings_without_nulls, counting_mask)
        pred_counting_mask = classifier_output["preds"]

        # Add predicted nulls to the original sentences.
        words_with_nulls = self._add_nulls(words, pred_counting_mask)
        return {"words": words_with_nulls, "loss": classifier_output["loss"]}

    @staticmethod
    def _prepend_cls(sentences: list[list[str]]) -> list[list[str]]:
        """
        Return a copy of sentences with [CLS] token prepended.
        """
        return [["[CLS]", *sentence] for sentence in sentences]

    @staticmethod
    def _remove_nulls(sentences: list[list[str]]) -> list[list[str]]:
        """
        Return a copy of sentences with nulls removed.
        """
        return [[word for word in sentence if word != "#NULL"] for sentence in sentences]

    @staticmethod
    def _add_nulls(sentences: list[list[str]], counting_mask: Tensor) -> list[list[str]]:
        """
        Return a copy of sentences with nulls restored according to counting masks.
        """
        sentences_with_nulls = []
        for sentence, counting_mask in zip(sentences, counting_mask):
            sentence_with_nulls = []
            for word, n_nulls_to_insert in zip(sentence, counting_mask):
                sentence_with_nulls.append(word)
                for _ in range(1, n_nulls_to_insert + 1):
                    sentence_with_nulls.append("#NULL")
            sentences_with_nulls.append(sentence_with_nulls)
        return sentences_with_nulls
