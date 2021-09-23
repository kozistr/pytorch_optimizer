from argparse import ArgumentParser, Namespace

from pylint.lint import Run


def get_configuration() -> Namespace:
    parser = ArgumentParser(prog='LINT')
    parser.add_argument(
        '-p',
        '--path',
        help='path to directory you want to run pylint | ' 'Default: %(default)s | ' 'Type: %(type)s ',
        default='pytorch_optimizer',
        type=str,
    )
    parser.add_argument(
        '-t',
        '--threshold',
        help='score threshold to fail pylint runner | ' 'Default: %(default)s | ' 'Type: %(type)s ',
        default=9.5,
        type=float,
    )

    return parser.parse_args()


def main():
    args: Namespace = get_configuration()

    path: str = str(args.path)
    threshold: float = float(args.threshold)
    print(f'PyLint Starting | path: {path} | threshold: {threshold:.2f}')

    results = Run([path], do_exit=False)

    final_score: float = results.linter.stats['global_note']
    if final_score < threshold:
        print(f'PyLint Failed | score: {final_score:.2f} | threshold: {threshold:.2f}')
        raise Exception
    else:
        print(f'PyLint Passed | score: {final_score:.2f} | threshold: {threshold:.2f}')


if __name__ == '__main__':
    main()
