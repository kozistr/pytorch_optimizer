from collections import defaultdict
from typing import Any, Callable, Dict, List, Type

from torch.optim import SGD, Adam, AdamW, Optimizer

__all__ = ['MuAdam', 'MuAdamW', 'MuSGD', 'get_mup_param_groups']

MU_P_MODES = ('adam', 'sgd')


def _get_infshape(p) -> Any:
    infshape = getattr(p, 'infshape', None)
    if infshape is None:
        raise ValueError(
            f'A parameter with shape {tuple(p.shape)} does not have `infshape` attribute. '
            'Did you forget to call `mup.set_base_shapes` on the model?'
        )
    return infshape


def _split_group(param_group: Dict[str, Any], mode: str):
    r"""Split one parameter group by the kind of each parameter and by its multiplier."""

    def new_group() -> Dict[str, Any]:
        return {**{k: v for k, v in param_group.items() if k != 'params'}, 'params': []}

    matrix_like = defaultdict(new_group)
    vector_like = defaultdict(new_group)
    fixed = new_group()

    for p in param_group['params']:
        infshape = _get_infshape(p)

        num_infinite: int = infshape.ninf()
        if num_infinite > 2:
            raise NotImplementedError('more than 2 inf dimensions')

        if num_infinite == 2:
            ratio = infshape.width_mult() if mode == 'adam' else infshape.fanin_fanout_mult_ratio()
            matrix_like[ratio]['params'].append(p)
        elif num_infinite == 1 and mode == 'sgd':
            vector_like[infshape.width_mult()]['params'].append(p)
        else:
            fixed['params'].append(p)

    return matrix_like, vector_like, fixed


def get_mup_param_groups(
    params,
    mode: str = 'adam',
    decoupled_wd: bool = False,
    **kwargs,
) -> List[Dict[str, Any]]:
    r"""Split parameter groups by their μP width multiplier and scale the learning rate and the weight decay.

    Every parameter must carry the `infshape` attribute, as set by `mup.set_base_shapes` of
    [microsoft/mup](https://github.com/microsoft/mup) (`ninf()`, `width_mult()` and `fanin_fanout_mult_ratio()` are
    the only methods used). The rules follow Table 8 of Tensor Programs V and the reference `mup.optim`.

    * `adam`: matrix-like weights (two infinite dimensions) use `lr / width_mult` and `weight_decay * width_mult`.
      Vector-like weights and the fixed weights keep the given values.
    * `sgd`: vector-like weights (one infinite dimension) use `lr * width_mult` and `weight_decay / width_mult`.
      Matrix-like weights use `lr / fanin_fanout_mult_ratio` and `weight_decay * fanin_fanout_mult_ratio`.
      Fixed weights keep the given values.

    Groups that end up with no parameter are not returned.

    Args:
        params: iterable of parameters or dicts defining parameter groups.
        mode (str): `adam` for optimizers that normalize the gradient entrywise (Adam, AdamW, RMSprop, Adagrad, ...) or
            `sgd` for SGD-like optimizers.
        decoupled_wd (bool): skip the scaling of the weight decay. Use it with an optimizer that does not multiply the
            weight decay by the learning rate.
        kwargs: `lr` and (optionally) `weight_decay`, the defaults for the groups that do not set them.

    """
    if mode not in MU_P_MODES:
        raise ValueError(f'mode {mode} must be one of ({" or ".join(MU_P_MODES)})')

    param_groups = list(params)
    if len(param_groups) == 0:
        raise ValueError('optimizer got an empty parameter list')
    if not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]

    new_param_groups: List[Dict[str, Any]] = []
    for param_group in param_groups:
        if 'lr' not in param_group:
            if 'lr' not in kwargs:
                raise ValueError('`lr` must be set either in the parameter group or as a keyword argument')
            param_group = {**param_group, 'lr': kwargs['lr']}
        if 'weight_decay' not in param_group:
            param_group = {**param_group, 'weight_decay': kwargs.get('weight_decay', 0.0)}

        matrix_like, vector_like, fixed = _split_group(param_group, mode)

        for ratio, group in matrix_like.items():
            group['lr'] /= ratio
            if not decoupled_wd:
                group['weight_decay'] *= ratio

        for width_mult, group in vector_like.items():
            group['lr'] *= width_mult
            if not decoupled_wd:
                group['weight_decay'] /= width_mult

        groups = [*matrix_like.values(), *vector_like.values(), fixed]
        new_param_groups.extend(group for group in groups if len(group['params']) > 0)

    return new_param_groups


def MuAdam(  # noqa: N802
    params,
    impl: Callable[..., Optimizer] = Adam,
    decoupled_wd: bool = False,
    **kwargs,
) -> Optimizer:
    r"""Adam with μP scaling.

    The model needs its base shapes set already, with `mup.set_base_shapes`. Any optimizer that normalizes the gradient
    entrywise can be passed as `impl`.

    Args:
        params: iterable of parameters or dicts defining parameter groups.
        impl (Callable): the Adam-like optimizer class.
        decoupled_wd (bool): skip the μP scaling of the weight decay.
        kwargs: parameters for `impl`. `lr` is required.

    Returns:
        An instance of `impl` with one parameter group per width multiplier.

    """
    return impl(get_mup_param_groups(params, mode='adam', decoupled_wd=decoupled_wd, **kwargs), **kwargs)


def MuAdamW(params, **kwargs) -> Optimizer:  # noqa: N802
    r"""AdamW with μP scaling. See `MuAdam`."""
    return MuAdam(params, impl=AdamW, **kwargs)


def MuSGD(  # noqa: N802
    params,
    impl: Type[Optimizer] = SGD,
    decoupled_wd: bool = False,
    **kwargs,
) -> Optimizer:
    r"""SGD with μP scaling.

    The model needs its base shapes set already, with `mup.set_base_shapes`. Any SGD-like optimizer can be passed as
    `impl`.

    Args:
        params: iterable of parameters or dicts defining parameter groups.
        impl (Callable): the SGD-like optimizer class.
        decoupled_wd (bool): skip the μP scaling of the weight decay.
        kwargs: parameters for `impl`. `lr` is required.

    Returns:
        An instance of `impl` with one parameter group per width multiplier.

    """
    return impl(get_mup_param_groups(params, mode='sgd', decoupled_wd=decoupled_wd, **kwargs), **kwargs)
