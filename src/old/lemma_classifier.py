from overrides import override

import re
import string

import torch
from torch import nn
from torch import Tensor, BoolTensor, LongTensor

from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN

from mlp import MLP
from lemmatize_helper import LemmaRule, predict_lemma_from_rule, normalize, DEFAULT_LEMMA_RULE


class LemmaClassifier(MLP):
    """MLP for lemma classification."""

    PUNCTUATION = set(string.punctuation)

    def __init__(
        self,
        vocab: Vocabulary,
        in_dim: int,
        hid_dim: int,
        n_classes: int,
        activation: str,
        dropout: float,
        dictionaries: list[dict[str, str]] = [],
        topk: int = None
    ):
        super().__init__(in_dim, hid_dim, n_classes, activation, dropout)

        self.vocab = vocab

        self.dictionary = set()
        for dictionary_info in dictionaries:
            dictionary_path = dictionary_info["path"]
            lemma_match_pattern = dictionary_info["lemma_match_pattern"]
            with open(dictionary_path, 'r') as f:
                txt = f.read()
                lemmas = re.findall(lemma_match_pattern, txt, re.MULTILINE)
                lemmas = set(map(normalize, lemmas))
            self.dictionary |= lemmas

        # If dictionary is given, topk must be set as well and vise versa.
        if self.dictionary:
            assert(topk is not None and topk >= 1)
        else:
            assert(topk is None)
        self.topk = topk

    @override
    def forward(
        self,
        embeddings: Tensor,
        labels: LongTensor = None,
        mask: BoolTensor = None,
        metadata: dict = None
    ) -> dict[str, Tensor]:

        output = super().forward(embeddings, labels, mask)
        preds, probs, loss = output['preds'], output['probs'], output['loss']
        # During the inference try to avoid malformed lemmas using external dictionary (if provided).
        if not self.training and self.dictionary:
            self._refine_predictions(preds, probs)
        return {'preds': preds, 'loss': loss}

    def _refine_predictions(self, probs: Tensor, preds: list[list[int]]):
        # Top-k most probable lemma rules for each token.
        top_rules = torch.topk(probs, k=self.topk, dim=-1).indices.cpu().numpy()

        for sentence_idx in range(len(metadata)):
            tokens = metadata[sentence_idx]
            tokens_top_rules = top_rules[sentence_idx]

            for token_idx in range(len(tokens)):
                token = str(tokens[token_idx])
                token_top_rules = tokens_top_rules[token_idx]

                # Lemmatizer usually does well with titles (e.g. 'Вася')
                # and different kind of dates (like '70-е')
                # so don't correct predictions in these cases.
                is_punctuation = lambda word: word in LemmaClassifier.PUNCTUATION
                is_title = lambda word: word[0].isupper()
                contains_digit = lambda word: any(char.isdigit() for char in word)
                if is_punctuation(token) or is_title(token) or contains_digit(token):
                    continue

                # Find the most probable correct lemma for the word.
                better_lemma_found = False
                for lemma_rule_id in token_top_rules:
                    lemma_rule_str = self.vocab.get_token_from_index(lemma_rule_id, "lemma_rule_labels")

                    # If the current most confident lemma rule is "unknown rule",
                    # then lemmatizer has no idea how to lemmatize the token, so interrupt lookup.
                    if lemma_rule_str == DEFAULT_OOV_TOKEN:
                        continue

                    lemma_rule = LemmaRule.from_str(lemma_rule_str)
                    lemma = predict_lemma_from_rule(token, lemma_rule)
                    if normalize(lemma) in self.dictionary:
                        better_lemma_found = True
                        break

                # Update predictions with better lemma.
                if better_lemma_found:
                    preds[sentence_idx][token_idx] = lemma_rule_id

