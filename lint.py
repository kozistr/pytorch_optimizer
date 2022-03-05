from argparse import ArgumentParser, Namespace

from pylint.lint import Run


def get_configuration() -> Namespace:
    parser = ArgumentParser(description='pylint configuration')
    parser.add_argument(
        '-p',
        '--path',
        default='pytorch_optimizer',
        type=str,
    )
    parser.add_argument(
        '-t',
        '--threshold',
        default=9.95,
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
        raise Exception(f'PyLint Failed | score {final_score:.2f} < threshold {threshold:.2f}')

    print(f'[+] PyLint Passed | score {final_score:.2f} > threshold {threshold:.2f}')


if __name__ == '__main__':
    main()
