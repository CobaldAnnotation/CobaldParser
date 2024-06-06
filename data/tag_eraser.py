#!/usr/bin/env python3

import sys
import argparse
from tqdm import tqdm

sys.path.append("..") # import 'common' package
from common.parse_conllu import parse_conllu, write_conllu
from common.sentence import Sentence


def main(
    input_file_path: str,
    output_file_path: str,
    erase_morphosyntax: bool,
    erase_semslot: bool,
    erase_semclass: bool
) -> None:

    with open(input_file_path, "r") as file:
        sentences = parse_conllu(file)

    for sentence in tqdm(sentences):
        for token in sentence:
            if erase_morphosyntax:
                token.lemma = None
                token.upos = None
                token.xpos = None
                token.feats = None
                token.head = None
                token.deprel = None
                token.deps = None
                token.misc = None
            if erase_semslot:
                token.semslot = None
            if erase_semclass:
                token.semclass = None
    write_conllu(output_file_path, sentences)


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
        '--erase-morphosyntax',
        action='store_true'
    )
    parser.add_argument(
        '--erase-semslot',
        action='store_true'
    )
    parser.add_argument(
        '--erase-semclass',
        action='store_true'
    )
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.erase_morphosyntax, args.erase_semslot, args.erase_semclass)

