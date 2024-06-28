#!/usr/bin/env python3

import sys
import argparse
from copy import deepcopy

from tqdm import tqdm

from typing import Dict, Iterable, List
from conllu.models import TokenList

sys.path.append('..')
from common.parse_conllu import parse_conllu_raw, write_conllu
from common.sentence import Sentence, Token


def is_range_token(token) -> bool:
    return '-' in token["id"]


def postprocess(orig_token_lists: List[TokenList], pred_token_lists: List[TokenList]):
    """
    Modifies pred_token_lists!
    """
    assert len(orig_token_lists) == len(pred_token_lists)
    for orig_token_list, pred_token_list in zip(orig_token_lists, pred_token_lists):
        assert orig_token_list.metadata["sent_id"] == pred_token_list.metadata["sent_id"]
        for orig_token in orig_token_list:
            if is_range_token(orig_token):
                range_token_start_id = orig_token["id"].split('-')[0]
                range_token_insert_pos = -1
                for pred_token_index, pred_token in enumerate(pred_token_list):
                    if pred_token["id"] == range_token_start_id:
                        range_token_insert_pos = pred_token_index
                        break
                assert 0 <= range_token_insert_pos
                pred_token_list.insert(range_token_insert_pos, orig_token)

    processed_sentences = []
    for pred_token_list in pred_token_lists:
        processed_sentences.append(Sentence.from_conllu(pred_token_list, renumerate=False))
    return processed_sentences


def main(original_file_path: str, predicted_file_path: str, output_file_path: str) -> None:
    print(f"Loading sentences...")
    with open(original_file_path, "r", encoding='utf8') as original_file:
        # Use parse_conllu_raw, because it preserves tags as-is (e.g. doesn't renumerate tokens ids)
        original_token_lists = parse_conllu_raw(original_file)
    with open(predicted_file_path, "r", encoding='utf8') as predicted_file:
        predicted_token_lists = parse_conllu_raw(predicted_file)
    print("Processing...")
    processed_sentences = postprocess(original_token_lists, predicted_token_lists)
    print("Writing results...")
    write_conllu(output_file_path, processed_sentences)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Insert range tokens (ones with x-y indices) from original sentences into predicted sentences, "
        "since parser doesn't work with range tokens."
    )
    parser.add_argument(
        'original_file',
        type=str,
        help='Original file with range tokens.'
    )
    parser.add_argument(
        'predicted_file',
        type=str,
        help='File with parser predictions without range tokens.'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Output file with range tokens.'
    )
    args = parser.parse_args()
    main(args.original_file, args.predicted_file, args.output_file)

