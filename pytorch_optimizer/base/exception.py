class NoSparseGradientError(Exception):
    r"""Raised when the gradient is sparse gradient.

    Args:
        optimizer_name (str): The name of the optimizer where the error occurred.
        note (str): Additional special conditions or notes (default is an empty string).
    """

    def __init__(self, optimizer_name: str, note: str = ''):
        self.note: str = ' ' if not note else f' w/ {note} '
        self.message: str = f'{optimizer_name}{self.note}does not support sparse gradient.'
        super().__init__(self.message)


class ZeroParameterSizeError(Exception):
    """Raised when the parameter size is 0."""

    def __init__(self):
        self.message: str = 'parameter size is 0'
        super().__init__(self.message)


class NoClosureError(Exception):
    """Raised when there's no closure function."""

    def __init__(self, optimizer_name: str, note: str = ''):
        self.message: str = f'{optimizer_name} requires closure.{note}'
        super().__init__(self.message)


class NegativeLRError(Exception):
    """Raised when learning rate is negative."""

    def __init__(self, lr: float, lr_type: str = ''):
        self.note: str = lr_type if lr_type else 'learning rate'
        self.message: str = f'{self.note} must be positive. ({lr} > 0)'
        super().__init__(self.message)


class NegativeStepError(Exception):
    """Raised when step is negative."""

    def __init__(self, num_steps: int, step_type: str = ''):
        self.note: str = step_type if step_type else 'step'
        self.message: str = f'{self.note} must be positive. ({num_steps} > 0)'
        super().__init__(self.message)


class NoComplexParameterError(Exception):
    r"""Raised when the dtype of the parameter is complex.

    Args:
        optimizer_name (str): The name of the optimizer where the error occurred.
        note (str): Additional special conditions or notes (default is an empty string).
    """

    def __init__(self, optimizer_name: str, note: str = ''):
        self.note: str = ' ' if not note else f' w/ {note} '
        self.message: str = f'{optimizer_name}{self.note}does not support complex parameter.'
        super().__init__(self.message)
