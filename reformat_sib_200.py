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


# Translations for category names
CATEGORY_MAPS = {
    'fin_Latn': {
        'entertainment': 'viihde',
        'geography': 'maantiede',
        'health': 'terveys',
        'politics': 'politiikka',
        'science/technology': 'tiede/teknologia',
        'sports': 'urheilu',
        'travel': 'matkailu',
    },
    'eng_Latn': {
        'entertainment': 'entertainment',
        'geography': 'geography',
        'health': 'health',
        'politics': 'politics',
        'science/technology': 'science/technology',
        'sports': 'sports',
        'travel': 'travel',
    }
}


def main(argv):
    args = argparser().parse_args(argv[1:])

    if args.lang_script not in CATEGORY_MAPS:
        print(f'No translations for {args.lang_script}', file=sys.stderr)
        return 1
    category_map = CATEGORY_MAPS[args.lang_script]

    source = load_dataset(args.source, args.lang_script)

    combined = concatenate_datasets([
        source['train'],
        source['validation'],
        source['test'],
    ])

    def translate_category(e):
        e["category"] = category_map[e["category"]]
        return e
    combined = combined.map(translate_category)

    categories = sorted(set(combined['category']))
    combined = combined.add_column('choices', [categories] * len(combined))

    answer_indices = [categories.index(c) for c in combined['category']]
    combined = combined.add_column('answer_idx', answer_indices)

    combined_dict = DatasetDict({ 'test': combined })
    combined_dict.push_to_hub(args.repo)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
