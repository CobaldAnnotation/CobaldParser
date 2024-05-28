import sys
import numpy as np
from sklearn.metrics import f1_score

from tqdm import tqdm

# zip `strict` is only available starting Python 3.10.
from more_itertools import zip_equal

from typing import Iterable, List, Tuple, Dict, Optional

from scorer.taxonomy import Taxonomy

sys.path.append("..") # import 'common' package
from common.token import Token
from common.sentence import Sentence


class CobaldScorer:
    def __init__(
        self,
        taxonomy_file: str,
        semclasses_out_of_taxonomy: set,
        lemma_weights: Dict[str, float] = None,
        feats_weights: Dict[str, float] = None
    ):
        self.taxonomy = Taxonomy(taxonomy_file)
        self.semclasses_out_of_taxonomy = set(semclasses_out_of_taxonomy)
        self.lemma_weights = lemma_weights
        self.feats_weights = feats_weights

    def score_lemma(self, test: Token, gold: Token) -> float:
        ignore_case_and_yo = lambda word: word.lower().replace('ё', 'е')
        score = ignore_case_and_yo(test.lemma) == ignore_case_and_yo(gold.lemma)

        if self.lemma_weights is not None:
            score *= self.lemma_weights[gold.upos]
        assert 0. <= score <= 1.
        return score

    def score_upos(self, test: Token, gold: Token) -> float:
        score = test.upos == gold.upos
        assert 0. <= score <= 1.
        return score

    def score_xpos(self, test: Token, gold: Token) -> float:
        score = test.xpos == gold.xpos
        assert 0. <= score <= 1.
        return score

    def score_feats(self, test: Token, gold: Token) -> float:
        if len(test.feats) == 0 or len(gold.feats) == 0:
            return len(test.feats) == len(gold.feats)

        correct_feats_weighted_sum = np.sum([
            (self.feats_weights[gram_cat] if self.feats_weights is not None else 1)
            * (gold.feats[gram_cat] == test.feats[gram_cat])
            for gram_cat in gold.feats
            if gram_cat in test.feats
        ])
        gold_feats_weighted_sum = np.sum([
            (self.feats_weights[gram_cat] if self.feats_weights is not None else 1)
            for gram_cat in gold.feats
        ])
        assert correct_feats_weighted_sum <= gold_feats_weighted_sum

        # Penalize test if it is longer than gold.
        # If there were no such penalty, one could simply predict all grammatical categories
        # existing for each token and score would not get any worse.
        # It's not what we expect from a good morphology classifier, so use penalty.
        penalty = 1 / (1 + max(len(test.feats) - len(gold.feats), 0))

        assert gold_feats_weighted_sum != 0
        score = penalty * correct_feats_weighted_sum / gold_feats_weighted_sum
        assert 0. <= score <= 1.
        return score

    def score_head(self, test: Token, gold: Token) -> float:
        score = test.head == gold.head
        assert 0. <= score <= 1.
        return score

    def score_deprel(self, test: Token, gold: Token) -> float:
        score = (test.head == gold.head) and (test.deprel == gold.deprel)
        assert 0. <= score <= 1.
        return score

    def score_deps_heads(self, test: Token, gold: Token) -> float:
        # Deps is a dict of str -> str.
        test_heads = set(test.deps.keys())
        gold_heads = set(gold.deps.keys())
        max_len = max(len(test_heads), len(gold_heads))
        score = len(test_heads & gold_heads) / max_len if 0 < max_len else 1.
        assert 0. <= score <= 1.
        return score

    def score_deps_rels(self, test: Token, gold: Token) -> float:
        # Deps is a dict of str -> str.
        test_deps = set(test.deps.items())
        gold_deps = set(gold.deps.items())
        score = 0.
        max_len = max(len(test_deps), len(gold_deps))
        score = len(test_deps & gold_deps) / max_len if 0 < max_len else 1.
        assert 0. <= score <= 1.
        return score

    def score_misc(self, test: Token, gold: Token) -> float:
        score = test.misc == gold.misc
        assert 0. <= score <= 1.
        return score

    def score_semslot(self, test: Token, gold: Token) -> float:
        score = test.semslot == gold.semslot
        assert 0. <= score <= 1.
        return score

    def score_semclass(self, test: Token, gold: Token) -> float:
        # Handle extra cases.
        if gold.semclass in self.semclasses_out_of_taxonomy:
            return test.semclass == gold.semclass

        # FIXME!
        if not self.taxonomy.has_semclass(gold.semclass):
            print(f"Unknown gold semclass encountered: {gold.semclass}")
            return 0.
        #assert self.taxonomy.has_semclass(gold.semclass), \
        #    f"Unknown gold semclass encountered: {gold.semclass}"

        if not self.taxonomy.has_semclass(test.semclass):
            return 0.

        semclasses_distance = self.taxonomy.calc_path_length(test.semclass, gold.semclass)

        # If distance is 0 then test_semclass == gold_semclass, so score is 1.
        # If they are different, the penalty is proportional to their distance.
        # If they are in different trees, then distance is inf, so score is 0.
        score = 1 / (1 + semclasses_distance)
        assert 0. <= score <= 1.
        return score

    def score_sentences(
        self,
        test_sentences: Iterable[Sentence],
        gold_sentences: Iterable[Sentence]
    ) -> Tuple[float]:
        # Grammatical scores.
        # Use 'list' and 'np.sum' instead of 'int' and '+=' for numerical stability.
        lemma_scores = []
        upos_scores = []
        xpos_scores = []
        feats_scores = []
        # Syntax scores (UAS, LAS, EUAS, ELAS).
        head_scores = []
        deprel_scores = []
        deps_heads_scores = []
        deps_rels_scores = []
        # Misc
        misc_scores = []
        # Semantic scores.
        semslot_scores = []
        semclass_scores = []
        # Nulls for f1 statistic.
        pred_nulls = []
        gold_nulls = []

        # 'lemma' scores are weighted.
        # Why? The idea here is quite natural:
        # We want immutable parts of speech (which are relatively easy to lemmatize) to affect
        # lemmatization score less than mutable ones (which are, obviously, harder to lemmatize).
        #
        # As a result, lemmatization per-token scores can be greater than 1.0
        # (for example, lemma_scores[10] can be equal to 10).
        #
        # However, we expect average dataset scores to be in [0.0..1.0] range,
        # so we also accumulate gold per-token scores and use them for
        # final normalization. This way we get 1.0 score if all test and gold lemmas are equal,
        # and a lower score otherwise.
        lemma_gold_scores = []

        for test_sentence, gold_sentence in tqdm(zip_equal(test_sentences, gold_sentences), file=sys.stdout):
            assert test_sentence.sent_id == gold_sentence.sent_id, \
                f"Test and gold sentence id mismatch at test_sentence.sent_id={test_sentence.sent_id}."

            # Test and gold sentence lengths may be different due to #NULL tokens.
            # If it is the case, insert extra "empty" tokens so that real tokens have same index in both senteces.
            test_sentence_aligned, gold_sentence_aligned = self._align_sentences(test_sentence, gold_sentence)
            assert len(test_sentence_aligned) == len(gold_sentence_aligned)

            for test_token, gold_token in zip_equal(test_sentence_aligned, gold_sentence_aligned):
                is_mismatched = test_token.is_empty() or gold_token.is_empty()

                assert test_token.form == gold_token.form or is_mismatched, \
                    f"Error at sent_id={test_sentence.sent_id} : Tokens forms are mismatched."

                # Score test_token.
                lemma_score = self.score_lemma(test_token, gold_token) if not is_mismatched else 0.
                upos_score = self.score_upos(test_token, gold_token) if not is_mismatched else 0.
                xpos_score = self.score_xpos(test_token, gold_token) if not is_mismatched else 0.
                feats_score = self.score_feats(test_token, gold_token) if not is_mismatched else 0.
                head_score = self.score_head(test_token, gold_token) if not is_mismatched else 0.
                deprel_score = self.score_deprel(test_token, gold_token) if not is_mismatched else 0.
                deps_heads_score = self.score_deps_heads(test_token, gold_token) if not is_mismatched else 0.
                deps_rels_score = self.score_deps_rels(test_token, gold_token) if not is_mismatched else 0.
                misc_score = self.score_misc(test_token, gold_token) if not is_mismatched else 0.
                semslot_score = self.score_semslot(test_token, gold_token) if not is_mismatched else 0.
                semclass_score = self.score_semclass(test_token, gold_token) if not is_mismatched else 0.

                # Accumulcate test scores.
                lemma_scores.append(lemma_score)
                upos_scores.append(upos_score)
                xpos_scores.append(xpos_score)
                feats_scores.append(feats_score)
                head_scores.append(head_score)
                deprel_scores.append(deprel_score)
                deps_heads_scores.append(deps_heads_score)
                deps_rels_scores.append(deps_rels_score)
                misc_scores.append(misc_score)
                semslot_scores.append(semslot_score)
                semclass_scores.append(semclass_score)
                pred_nulls.append(test_token.is_null())
                gold_nulls.append(gold_token.is_null())

                # Score gold.
                lemma_gold_score = self.score_lemma(gold_token, gold_token) if not is_mismatched else 1.

                # Accumulate gold scores.
                lemma_gold_scores.append(lemma_gold_score)

        # Average per-token scores over all tokens in all sentences.
        # Note that we cannot just average lemma_scores, for they are weighted.
        lemma_avg_score = np.mean(lemma_scores) / np.mean(lemma_gold_scores)
        upos_avg_score = np.mean(upos_scores)
        xpos_avg_score = np.mean(xpos_scores)
        feats_avg_score = np.mean(feats_scores)
        head_avg_score = np.mean(head_scores)
        deprel_avg_score = np.mean(deprel_scores)
        deps_heads_avg_score = np.mean(deps_heads_scores)
        deps_rels_avg_score = np.mean(deps_rels_scores)
        misc_avg_score = np.mean(misc_scores)
        semslot_avg_score = np.mean(semslot_scores)
        semclass_avg_score = np.mean(semclass_scores)
        null_f1 = f1_score(gold_nulls, pred_nulls)

        assert 0. <= lemma_avg_score <= 1.
        assert 0. <= upos_avg_score <= 1.
        assert 0. <= xpos_avg_score <= 1.
        assert 0. <= feats_avg_score <= 1.
        assert 0. <= head_avg_score <= 1.
        assert 0. <= deprel_avg_score <= 1.
        assert 0. <= deps_heads_avg_score <= 1.
        assert 0. <= deps_rels_avg_score <= 1.
        assert 0. <= misc_avg_score <= 1.
        assert 0. <= semslot_avg_score <= 1.
        assert 0. <= semclass_avg_score <= 1.
        assert 0. <= null_f1 <= 1.

        return (
            lemma_avg_score,
            upos_avg_score,
            xpos_avg_score,
            feats_avg_score,
            head_avg_score,
            deprel_avg_score,
            deps_heads_avg_score,
            deps_rels_avg_score,
            misc_avg_score,
            semslot_avg_score,
            semclass_avg_score,
            null_f1
        )

    def _align_sentences(self, lhs: Sentence, rhs: Sentence):
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
            if lhs[i].form == rhs[j].form:
                lhs_aligned.append(lhs[i])
                rhs_aligned.append(rhs[j])
                i += 1
                j += 1
            else:
                if lhs[i].is_null():
                    lhs_aligned.append(lhs[i])
                    rhs_aligned.append(Token.create_empty(id=f"{j}.1"))
                    i += 1
                else:
                    assert rhs[j].is_null()
                    lhs_aligned.append(Token.create_empty(id=f"{i}.1"))
                    rhs_aligned.append(rhs[j])
                    j += 1

        if i < len(lhs):
            # lhs has extra #NULLs at the end, so append #EMPTY node to rhs
            assert j == len(rhs)
            lhs_aligned.append(lhs[i])
            rhs_aligned.append(Token.create_empty(id=f"{j}.1"))
            i += 1
            assert i == len(lhs)
        if j < len(rhs):
            assert i == len(lhs)
            lhs_aligned.append(Token.create_empty(id=f"{i}.1"))
            rhs_aligned.append(rhs[j])
            j += 1
            assert j == len(rhs)

        assert len(lhs_aligned) == len(rhs_aligned)
        return Sentence(lhs_aligned, lhs.metadata), Sentence(rhs_aligned, rhs.metadata)

