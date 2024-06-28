#!/usr/bin/env python3

import sys
import argparse
from tqdm import tqdm

sys.path.append("..") # import 'common' package
from common.parse_conllu import parse_conllu_raw, write_conllu
from common.sentence import Token, Sentence


def main(
    input_file_path: str,
    output_file_path: str,
    erase_morphosyntax: bool,
    erase_semslot: bool,
    erase_semclass: bool
) -> None:

    with open(input_file_path, "r") as file:
        token_lists = parse_conllu_raw(file)

    clean_sentences = []
    for token_list in tqdm(token_lists):
        clean_tokens = []
        for token in token_list:
            if token["form"] == "#NULL":
                continue
            if erase_morphosyntax:
                token["lemma"] = None
                token["upos"] = None
                token["xpos"] = None
                token["feats"] = None
                token["head"] = None
                token["deprel"] = None
                token["deps"] = None
                token["misc"] = None
            if erase_semslot:
                token["semslot"] = None
            if erase_semclass:
                token["semclass"] = None
            # Skip range tokens.
            clean_tokens.append(Token(**token))
        clean_sentences.append(Sentence(clean_tokens, token_list.metadata, renumerate=False))

    write_conllu(output_file_path, clean_sentences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Remove tags from conllu file, leaving `id` and `form` intact.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Input conllu file.'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Output conllu file with tags removed.'
    )
    parser.add_argument(
        '--keep-morphosyntax',
        action='store_false'
    )
    parser.add_argument(
        '--keep-semslot',
        action='store_false'
    )
    parser.add_argument(
        '--keep-semclass',
        action='store_false'
    )
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.keep_morphosyntax, args.keep_semslot, args.keep_semclass)

