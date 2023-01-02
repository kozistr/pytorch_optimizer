class NoSparseGradientError(Exception):
    """Raised when the gradient is sparse gradient

    :param optimizer_name: str. name of the optimizer.
    """

    def __init__(self, optimizer_name: str, note: str = ''):
        self.note: str = ' ' if note == '' else f' w/ {note} '
        self.message: str = f'[-] {optimizer_name}{self.note}does not support sparse gradient.'
        super().__init__(self.message)
