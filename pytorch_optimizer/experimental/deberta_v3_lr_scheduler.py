from torch import nn

from pytorch_optimizer.base.types import PARAMETERS


def deberta_v3_large_lr_scheduler(
    model: nn.Module,
    head_param_start: int = 390,
    base_lr: float = 2e-5,
    head_lr: float = 1e-4,
    wd: float = 1e-2,
) -> PARAMETERS:
    """DeBERTa-v3 large layer-wise lr scheduler
    Reference : https://github.com/gilfernandes/commonlit

    :param model: nn.Module. model. based on Huggingface Transformers.
    :param head_param_start: int. where the backbone ends (head starts)
    :param base_lr: float. base lr
    :param head_lr: float. head_lr
    :param wd: float. weight decay
    """
    named_parameters = list(model.named_parameters())

    backbone_parameters = named_parameters[:head_param_start]
    regressor_parameters = named_parameters[head_param_start:]

    regressor_group = [params for (_, params) in regressor_parameters]

    parameters = [{'params': regressor_group, 'lr': head_lr}]

    layer_low_threshold: int = 195  # start of the 12 layers
    layer_middle_threshold: int = 323  # end of the 24 layers

    for layer_num, (name, params) in enumerate(backbone_parameters):
        weight_decay: float = 0.0 if ('bias' in name) or ('LayerNorm.weight' in name) else wd

        lr = base_lr / 2.5  # 2e-5
        if layer_num >= layer_middle_threshold:
            lr = base_lr / 0.5  # 1e-4
        elif layer_num >= layer_low_threshold:
            lr = base_lr

        parameters.append({'params': params, 'weight_decay': weight_decay, 'lr': lr})

    return parameters
