import sys
import argparse
from more_itertools import zip_equal
from tqdm import tqdm

import numpy as np

from typing import Dict, Tuple

from scorer.scorer import CobaldScorer
from common.parse_conllu import parse_conllu_incr


def main(
    test1_file_path: str,
    test2_file_path: str,
    gold_file_path: str,
) -> Tuple[float]:

    scorer = CobaldScorer()

    test1_deps_scores = []
    test2_deps_scores = []

    with open(test1_file_path, 'r') as test1_file, open(test2_file_path, 'r') as test2_file, open(gold_file_path, 'r') as gold_file:
        test1_sentences = parse_conllu_incr(test1_file)
        test2_sentences = parse_conllu_incr(test2_file)
        gold_sentences = parse_conllu_incr(gold_file)

        for test1_sentence, test2_sentence, gold_sentence in tqdm(zip_equal(test1_sentences, test2_sentences, gold_sentences), file=sys.stdout):
            # Align three sentences.
            _, gold_sentence_aligned1 = scorer._align_sentences(test1_sentence, gold_sentence)
            _, gold_sentence_aligned2 = scorer._align_sentences(test2_sentence, gold_sentence)
            _, gold_sentence_aligned = scorer._align_sentences(gold_sentence_aligned1, gold_sentence_aligned2)
            test1_sentence_aligned, gold_sentence_aligned = scorer._align_sentences(test1_sentence, gold_sentence_aligned)
            test2_sentence_aligned, gold_sentence_aligned = scorer._align_sentences(test2_sentence, gold_sentence_aligned)
            assert len(test1_sentence_aligned) == len(test2_sentence_aligned) == len(gold_sentence_aligned), \
                f"{test1_sentence_aligned.serialize()}\n{test1_sentence_aligned.serialize()}\n{gold_sentence_aligned.serialize()}"

            sentence_is_printed = False
            for test1_token, test2_token, gold_token in zip_equal(test1_sentence_aligned, test2_sentence_aligned, gold_sentence_aligned):
                is_mismatched = test1_token.is_empty() or test2_token.is_empty() or gold_token.is_empty()
                assert test1_token.form == test2_token.form == gold_token.form or is_mismatched, \
                    f"Error at sent_id={test_sentence.sent_id} : Tokens forms are mismatched."

                if is_mismatched:
                    continue

                test1_deps_score = scorer.score_deps_rels(test1_token, gold_token)
                test2_deps_score = scorer.score_deps_rels(test2_token, gold_token)
                test1_deps_scores.append(test1_deps_score)
                test2_deps_scores.append(test2_deps_score)

                if test1_deps_score != test2_deps_score:
                    if not sentence_is_printed:
                        print(f"sent_id={gold_sentence.metadata['sent_id']}")
                        print(gold_sentence.metadata['text'])
                        print('-------------------------------')
                        sentence_is_printed = True
                    print(gold_token.form)
                    print(f"\tdeps score no semantic = {test1_deps_score:.2f} < {test2_deps_score:.2f} = deps score with semantic")
                    max_deps_char_len = max(map(lambda x: len(str(x)), [test1_token.deps, test2_token.deps, gold_token.deps]))
                    max_semslot_char_len = max(map(lambda x: len(str(x)), [test1_token.semslot, test2_token.semslot, gold_token.semslot]))
                    max_semclass_char_len = max(map(lambda x: len(str(x)), [test1_token.semclass, test2_token.semclass, gold_token.semclass]))
                    print(f"\tsem- deps={str(test1_token.deps):{max_deps_char_len}} SS={str(test1_token.semslot):{max_semslot_char_len}} SC={str(test1_token.semclass):{max_semclass_char_len}}")
                    print(f"\tsem+ deps={str(test2_token.deps):{max_deps_char_len}} SS={str(test2_token.semslot):{max_semslot_char_len}} SC={str(test2_token.semclass):{max_semclass_char_len}}")
                    print(f"\tgold deps={str(gold_token.deps):{max_deps_char_len}} SS={str(gold_token.semslot):{max_semslot_char_len}} SC={str(gold_token.semclass):{max_semclass_char_len}}")
            if sentence_is_printed:
                print('===============================\n')

        print(f"sem- deps score: {np.mean(test1_deps_scores)}")
        print(f"sem+ deps score: {np.mean(test2_deps_scores)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CoBaLD deps diff script.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'preds_no_semantic',
        type=str,
        help='Conllu file with predictions for parser trained without semantics.'
    )
    parser.add_argument(
        'preds_with_semantic',
        type=str,
        help='Conllu file with predictions for parser trained with semantics.'
    )
    parser.add_argument(
        'gold_file',
        type=str,
        help="Gold file with true tags."
    )
    args = parser.parse_args()

    main(args.preds_no_semantic, args.preds_with_semantic, args.gold_file)

