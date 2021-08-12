from argparse import ArgumentParser, Namespace


def parse() -> Namespace:
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--gpus', type=str, help='index of gpus. if multiple, use comma to list.'
    )
    args = parser.parse_args()
    return args
