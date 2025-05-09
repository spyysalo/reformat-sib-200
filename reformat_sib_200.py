#!/usr/bin/env python3

import sys

from argparse import ArgumentParser

from datasets import load_dataset, concatenate_datasets, DatasetDict


def argparser():
    ap = ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('lang_script')
    ap.add_argument('repo')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    source = load_dataset(args.source, args.lang_script)

    combined = concatenate_datasets([
        source['train'],
        source['validation'],
        source['test'],
    ])

    categories = sorted(set(combined['category']))
    combined = combined.add_column('choices', [categories] * len(combined))

    answer_indices = [categories.index(c) for c in combined['category']]
    combined = combined.add_column('answer_idx', answer_indices)

    combined_dict = DatasetDict({ 'test': combined })
    combined_dict.push_to_hub(args.repo)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
