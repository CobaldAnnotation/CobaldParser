import os
import sys
import argparse
import json

import numpy as np

from typing import Dict, Tuple

from scorer.scorer import CobaldScorer
from common.parse_conllu import parse_conllu_incr


OUTPUT_PRECISION = 4


def load_dict_from_json(json_file_path: str) -> Dict:
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return data


def main(
    test_file_path: str,
    gold_file_path: str,
    taxonomy_file: str,
    lemma_weights_file: str,
    feats_weights_file: str,
) -> Tuple[float]:

    print(f"Load taxonomy from {taxonomy_file}.")
    print(f"Load lemma weights from {lemma_weights_file}.")
    lemma_weights = load_dict_from_json(lemma_weights_file)
    print(f"Load feats weights from {feats_weights_file}.")
    feats_weights = load_dict_from_json(feats_weights_file)

    print("Build scorer...")
    scorer = CobaldScorer(
        taxonomy_file,
        semclasses_out_of_taxonomy={'_'},
        lemma_weights=lemma_weights,
        feats_weights=feats_weights
    )

    print("Evaluate...")
    with open(test_file_path, 'r') as test_file, open(gold_file_path, 'r') as gold_file:
        #feats_set = {}
        #for sent in parse_conllu_incr(test_file):
        #    for token in sent:
        #        feats = token.feats
        #        if feats is not None:
        #            for gram_cat, grammeme in feats.items():
        #                if gram_cat not in feats_set:
        #                    feats_set[gram_cat] = set()
        #                feats_set[gram_cat].add(grammeme)
        #print(f"feats: {feats_set}")
        test_sentences = parse_conllu_incr(test_file)
        gold_sentences = parse_conllu_incr(gold_file)
        scores = scorer.score_sentences(test_sentences, gold_sentences)

    # Exit on errors.
    if scores is None:
        print("Errors encountered, exit.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    lemma, upos, xpos, feats, uas, las, euas, elas, misc, semslot, semclass, null_f1 = scores
    # Average average scores into total score.
    total = np.mean([lemma, upos, xpos, feats, uas, las, euas, elas, misc, semslot, semclass, null_f1])

    return lemma, upos, xpos, feats, uas, las, euas, elas, misc, semslot, semclass, null_f1, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CoBaLD evaluation script.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'test_file',
        type=str,
        help='Test file in CoBaLD format with predicted tags.'
    )
    parser.add_argument(
        'gold_file',
        type=str,
        help="Gold file in CoBaLD format with true tags.\n"
        "For example, train.conllu."
    )
    script_dir = os.path.dirname(__file__)
    default_rel_taxonomy_path = "semantic-hierarchy/hyperonims_hierarchy.csv"
    parser.add_argument(
        '-taxonomy_file',
        type=str,
        help="File in CSV format with semantic class taxonomy.",
        default=os.path.normpath(os.path.join(script_dir, default_rel_taxonomy_path))
    )
    default_rel_lemma_weights_path = "scorer/weights_estimator/weights/lemma_weights.json"
    parser.add_argument(
        '-lemma_weights_file',
        type=str,
        help="JSON file with 'POS' -> 'lemma weight for this POS' relations.",
        default=os.path.normpath(os.path.join(script_dir, default_rel_lemma_weights_path))
    )
    default_rel_feats_weights_path = "scorer/weights_estimator/weights/feats_weights.json"
    parser.add_argument(
        '-feats_weights_file',
        type=str,
        help="JSON file with 'grammatical category' -> 'weight of this category' relations.",
        default=os.path.normpath(os.path.join(script_dir, default_rel_feats_weights_path))
    )
    args = parser.parse_args()

    lemma, upos, xpos, feats, uas, las, euas, elas, misc, semslot, semclass, null_f1, total = main(
        args.test_file,
        args.gold_file,
        args.taxonomy_file,
        args.lemma_weights_file,
        args.feats_weights_file,
    )

    print()
    print(f"======== SCORES =========")
    print(f"Lemmatization: {lemma:.{OUTPUT_PRECISION}f}")
    print(f"UPOS: {upos:.{OUTPUT_PRECISION}f}")
    print(f"XPOS: {xpos:.{OUTPUT_PRECISION}f}")
    print(f"Feats: {feats:.{OUTPUT_PRECISION}f}")
    print(f"UAS: {uas:.{OUTPUT_PRECISION}f}")
    print(f"LAS: {las:.{OUTPUT_PRECISION}f}")
    print(f"EUAS: {euas:.{OUTPUT_PRECISION}f}")
    print(f"ELAS: {elas:.{OUTPUT_PRECISION}f}")
    print(f"Misc: {misc:.{OUTPUT_PRECISION}f}")
    print(f"SemSlot: {semslot:.{OUTPUT_PRECISION}f}")
    print(f"SemClass: {semclass:.{OUTPUT_PRECISION}f}")
    print(f"Null F1: {null_f1:.{OUTPUT_PRECISION}f}")
    print(f"------------------------")
    print(f"Total: {total:.{OUTPUT_PRECISION}f}")

