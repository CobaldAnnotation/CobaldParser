"""
Based on https://github.com/DanAnastasyev/GramEval2020/blob/master/solution/train/lemmatize_helper.py
"""

from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class LemmaRule:
    cut_prefix: int = 0
    cut_suffix: int = 0
    append_suffix: str = ''

    @staticmethod
    def from_str(lemma_rule_str: str):
        rules = [rule.split('=')[-1] for rule in lemma_rule_str.split('|')]
        return LemmaRule(
            cut_prefix = int(rules[0]),
            cut_suffix = int(rules[1]),
            append_suffix = str(rules[2])
        )

    def __str__(self) -> str:
        return f"cut_prefix={self.cut_prefix}" + "|" + \
               f"cut_suffix={self.cut_suffix}" + "|" + \
               f"append_suffix={self.append_suffix}"


DEFAULT_LEMMA_RULE = LemmaRule()


def normalize(word: str) -> str:
    """
    Normalize word: cast to lowercase and replace russian 'ё' with 'е'
    """
    return word.lower().replace('ё', 'е')


def construct_lemma_rule(word: str, lemma: str) -> str:
    """
    Predict lemmatization rule given word and its lemma.
    Example:
    >>> construct_lemma_rule("сек.", "секунда")
    LemmaRule(cut_prefix=0, cut_suffix=1, append_suffix='унда')
    """
    word = normalize(word)
    lemma = normalize(lemma)

    match = SequenceMatcher(None, word, lemma).find_longest_match(0, len(word), 0, len(lemma))

    lemma_rule = LemmaRule(
        cut_prefix = match.a,
        cut_suffix = len(word) - (match.a + match.size),
        append_suffix = lemma[match.b + match.size:]
    )
    return str(lemma_rule)

def reconstruct_lemma(word: str, rule_str: str) -> str:
    rule = LemmaRule.from_str(rule_str)
    lemma = word[rule.cut_prefix:]
    lemma = lemma[:-rule.cut_suffix] if rule.cut_suffix != 0 else lemma
    lemma += rule.append_suffix
    return lemma
