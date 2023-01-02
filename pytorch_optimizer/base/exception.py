class NoSparseGradientError(Exception):
    """Raised when the gradient is sparse gradient

    :param optimizer_name: str. name of the optimizer.
    """

    def __init__(self, optimizer_name: str, note: str = ''):
        self.note: str = ' ' if note == '' else f' w/ {note} '
        self.message: str = f'[-] {optimizer_name}{self.note}does not support sparse gradient.'
        super().__init__(self.message)


class ZeroParameterSize(Exception):
    """Raised when the parameter size is 0"""

    def __init__(self):
        self.message: str = '[-] parameter size is 0'
        super().__init__(self.message)
