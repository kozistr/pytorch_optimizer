class NoSparseGradientError(Exception):
    """Raised when the gradient is sparse gradient

    :param optimizer_name: str. name of the optimizer.
    """
    def __init__(self, optimizer_name: str):
        self.message: str = f'[-] {optimizer_name} does not support sparse gradient.'
        super().__init__(self.message)
